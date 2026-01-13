import torch
import spmv  # Your custom extension
import time
import math
import os
import argparse

# --- Matrix Generators ---

def generate_ml_block_sparse_spd(rows, cols, block_size=32, density=0.05):
    """
    Generates a Block-Sparse SPD Matrix.
    Representative of 'Structured Pruning' in ML models (e.g. Transformers).
    
    Structure: The matrix is composed of dense BxB blocks.
    SPD Property: A = M^T @ M + alpha*I
    """
    print(f"\n[ML-Style] Generating Block Sparse ({rows}x{cols}, BlockSize={block_size}, Density={density})...")
    
    # 1. Create a Block Mask
    n_blocks_row = rows // block_size
    n_blocks_col = cols // block_size
    
    # Create a small mask for the blocks
    block_mask = torch.rand(n_blocks_row, n_blocks_col) < density
    
    # Upscale the mask to full size
    mask_rows = block_mask.repeat_interleave(block_size, dim=0)
    mask_full = mask_rows.repeat_interleave(block_size, dim=1)
    
    # Trim if dimensions don't divide perfectly
    mask_full = mask_full[:rows, :cols]
    
    # 2. Generate random values and apply mask
    # We generate a smaller random dense matrix M
    M = torch.randn(rows, cols) * mask_full.float()
    
    # 3. Make SPD: A = M^T @ M
    # Note: resulting density will increase slightly due to overlap in matmul
    print("  -> Computing A = M^T @ M (this creates the SPD structure)...")
    A_dense = torch.matmul(M.t(), M)
    
    # 4. Regularize diagonal (ensure positive definiteness)
    A_dense.diagonal().add_(1.0)
    
    # 5. sparsify
    A_sparse = A_dense.to_sparse()
    print(f"  -> Actual Final Density: {A_sparse._nnz() / (rows*cols):.4f}")
    
    return A_sparse, A_dense

def generate_physics_stencil_spd(grid_size):
    """
    Generates a 5-point Laplacian Stencil on a 2D grid.
    Representative of Physics/CFD simulations (Poisson Equation).
    
    Matrix Size: N = grid_size * grid_size
    Structure: Strictly Band Diagonal (Main diag + offsets).
    """
    N = grid_size * grid_size
    print(f"\n[Physics-Style] Generating Laplacian Stencil (Grid {grid_size}x{grid_size} -> Matrix {N}x{N})...")
    
    # Main diagonal (4.0)
    diagonals = [torch.full((N,), 4.0)]
    offsets = [0]
    
    # Off-diagonals (-1.0)
    # 1. Left/Right neighbors
    off_1 = torch.full((N-1,), -1.0)
    # Fix boundary conditions (periodic breaks in grid)
    for i in range(1, grid_size):
        off_1[i*grid_size - 1] = 0
    diagonals.extend([off_1, off_1])
    offsets.extend([1, -1])
    
    # 2. Top/Bottom neighbors
    off_grid = torch.full((N - grid_size,), -1.0)
    diagonals.extend([off_grid, off_grid])
    offsets.extend([grid_size, -grid_size])
    
    # Construct Sparse Matrix from Diagonals
    # Note: torch.sparse doesn't have a direct 'diags' constructor like scipy,
    # so we build it via dense or indices. For speed here, we use dense intermediate 
    # if N is small, or indices if N is large. Let's use indices construction for safety.
    
    indices_list = []
    values_list = []
    
    for diag, offset in zip(diagonals, offsets):
        # Create row indices
        if offset >= 0:
            rows = torch.arange(0, N - offset)
            cols = torch.arange(offset, N)
        else:
            rows = torch.arange(-offset, N)
            cols = torch.arange(0, N + offset)
            
        # Add to list
        indices_list.append(torch.stack([rows, cols]))
        values_list.append(diag)
        
    all_indices = torch.cat(indices_list, dim=1)
    all_values = torch.cat(values_list)
    
    A_sparse = torch.sparse_coo_tensor(all_indices, all_values, (N, N))
    A_dense = A_sparse.to_dense() # Needed for validation
    
    return A_sparse, A_dense

# --- Utilities ---

def pack_keys(row_indices, col_indices):
    row_indices = row_indices.to(torch.int64)
    col_indices = col_indices.to(torch.int64)
    return (row_indices << 32) | col_indices

def conjugate_gradient_torch(A, b, max_iter=100, tol=1e-4):
    x = torch.zeros_like(b)
    r = b - torch.matmul(A, x)
    p = r.clone()
    rsold = torch.dot(r, r)
    
    for i in range(1000):
        Ap = torch.matmul(A, p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

# --- Main Benchmark ---

def run_benchmark(mode, size, density):
    if mode == "ml":
        A_sparse, A_dense = generate_ml_block_sparse_spd(size, size, block_size=32, density=density)
    elif mode == "physics":
        # For physics, 'size' is the grid edge, so matrix dim is size*size
        # We approximate 'size' to be sqrt(N)
        grid_size = int(math.sqrt(size))
        A_sparse, A_dense = generate_physics_stencil_spd(grid_size)
    
    rows, cols = A_dense.shape
    indices = A_sparse.coalesce().indices()
    values = A_sparse.coalesce().values().to(torch.float32)
    nnz = values.shape[0]
    
    print(f"Matrix Ready. Shape: {rows}x{cols}, NNZ: {nnz}")

    # Prepare Data
    x_true = torch.randn(rows, dtype=torch.float32)
    # Normalize x_true to avoid exploding values
    x_true = x_true / torch.norm(x_true)
    b = torch.matmul(A_dense, x_true)
    
    # Custom Extension Setup
    keys = pack_keys(indices[0], indices[1])
    row_ptr = torch.zeros(rows + 1, dtype=torch.int32)
    col_ind = torch.zeros(nnz, dtype=torch.int32)
    val_buf = torch.zeros(nnz, dtype=torch.float32)
    x_sol = torch.zeros(rows, dtype=torch.float32)
    
    print("Initializing Custom Tensor...")
    start_init = time.time()
    tensor_impl = spmv.csr_tensor(keys, values, row_ptr, col_ind, val_buf, rows, cols)
    print(f"Init Time: {(time.time() - start_init)*1000:.2f} ms")

    # Benchmark Custom Solver
    print("\n>>> Benchmarking Custom CGD Solver...")
    
    # Warmup
    tensor_impl.iter_solve(b, x_sol)
    
    timings = []
    for _ in range(10):
        x_sol.zero_()
        t0 = time.time()
        tensor_impl.iter_solve(b, x_sol)
        timings.append(time.time() - t0)
        
    avg_custom = sum(timings) / len(timings)
    print(f"Avg Custom Time: {avg_custom*1000:.2f} ms")
    
    # Accuracy Check
    b_pred = torch.matmul(A_dense, x_sol)
    resid = torch.norm(b_pred - b) / (torch.norm(b) + 1e-8)
    print(f"Relative Residual: {resid:.6e}")
    
    # Benchmark PyTorch Dense
    print("\n>>> Benchmarking PyTorch Dense CG (CPU)...")
    t0 = time.time()
    conjugate_gradient_torch(A_dense, b, max_iter=200) # Match iter count roughly
    py_time = time.time() - t0
    print(f"PyTorch Dense Time: {py_time*1000:.2f} ms")
    
    print(f"\n>>> Speedup: {py_time / avg_custom:.2f}x")

if __name__ == "__main__":
    os.system("pip install . --no-build-isolation") 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["ml", "physics"], default="ml")
    parser.add_argument("--size", type=int, default=18976) # Matrix row count
    parser.add_argument("--density", type=float, default=0.015)
    args = parser.parse_args()
    
    run_benchmark(args.mode, args.size, args.density)