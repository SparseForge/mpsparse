import torch
import spmv  # The name defined in your setup.py
import time
import math
import os
import argparse

def generate_spd_matrix(rows, cols, density=0.01):
    """
    Generates a Sparse Symmetric Positive Definite matrix.
    Method: A = M^T * M + alpha * I
    """
    print(f"Generating random sparse matrix ({rows}x{cols}, density={density})...")
    
    # 1. Generate a random sparse matrix M
    nnz = int(rows * cols * density)
    indices = torch.randint(0, rows, (2, nnz))
    values = torch.rand(nnz)
    M = torch.sparse_coo_tensor(indices, values, (rows, cols))
    
    # 2. Make it dense temporarily to perform A = M^T @ M
    M_dense = M.to_dense()
    A_dense = torch.matmul(M_dense.t(), M_dense)
    
    # 3. Add diagonal regularization to ensure Positive Definiteness
    # Increased to 5.0 to make the matrix better conditioned (more stable)
    A_dense.diagonal().add_(5.0)
    
    # 4. Convert back to sparse COO
    A_sparse = A_dense.to_sparse()
    
    print("Matrix generation complete.")
    return A_sparse, A_dense

def pack_keys(row_indices, col_indices):
    """
    Packs row and column indices into a single uint64 tensor.
    """
    row_indices = row_indices.to(torch.int64)
    col_indices = col_indices.to(torch.int64)
    packed = (row_indices << 32) | col_indices
    return packed

def benchmark(n_rows=40108, density=0.05):
    # --- 1. Data Preparation ---
    A_sparse, A_dense = generate_spd_matrix(n_rows, n_rows, density)
    
    indices = A_sparse.indices()
    values = A_sparse.values().to(torch.float32)
    
    row_indices = indices[0]
    col_indices = indices[1]
    nnz = values.shape[0]

    # Create target vector 'b'
    x_true = torch.randn(n_rows, dtype=torch.float32)
    b = torch.matmul(A_dense, x_true)

    # Preparce inputs for C++ Extension
    keys = pack_keys(row_indices, col_indices)
    row_ptr_buffer = torch.zeros(n_rows + 1, dtype=torch.int32)
    col_ind_buffer = torch.zeros(nnz, dtype=torch.int32)
    out_vals_buffer = torch.zeros(nnz, dtype=torch.float32)
    x_sol = torch.zeros(n_rows, dtype=torch.float32)

    print(f"\nInitializing Metal CSR Tensor (N={n_rows}, NNZ={nnz})...")
    
    # --- 2. Initialize Extension ---
    start_init = time.time()
    tensor_impl = spmv.csr_tensor(
        keys,
        values,
        row_ptr_buffer,
        col_ind_buffer,
        out_vals_buffer,
        n_rows,
        n_rows
    )
    end_init = time.time()
    print(f"Initialization (COO->CSR on GPU) took: {(end_init - start_init)*1000:.2f} ms")

    # --- 3. Run Benchmark ---
    print("\nStarting Solver Benchmark...")
    
    # Warmup
    x_warmup = torch.zeros_like(x_sol)
    tensor_impl.iter_solve(b, x_warmup)
    
    num_runs = 10
    timings = []
    
    for i in range(num_runs):
        x_sol.zero_()
        t0 = time.time()
        tensor_impl.iter_solve(b, x_sol)
        t1 = time.time()
        timings.append((t1 - t0) * 1000)
        print("b_loop", b)
        print("x_sol", x_sol)

    avg_time = sum(timings) / len(timings)
    print(f"Average Solve Time (custom CGD): {avg_time:.2f} ms")

    # --- 4. Validation ---
    
    # CHECK 1: Did the solver explode?
    if torch.isnan(x_sol).any() or torch.isinf(x_sol).any():
        print("\n❌ CRITICAL: The solver output contains NaNs or Infs.")
        print("   -> This means the solver diverged (exploded).")
        print("   -> Try increasing diagonal regularization or checking C++ dot-product logic.")
    else:
        # CHECK 2: Calculate Residual with Safe Division
        b_pred = torch.matmul(A_dense, x_sol)
        print(b_pred)
        print(b)
        print(torch.norm(b_pred - b))
        
        # Add epsilon (1e-8) to denominator to prevent division by zero
        epsilon = 1e-8
        numerator = torch.norm(b_pred - b)
        denominator = torch.norm(b) + epsilon
        
        residual = numerator / denominator
        
        print(f"\n--- Results ---")
        print(f"Numerator (Error Norm): {numerator:.6e}")
        print(f"Denominator (Target Norm): {denominator:.6e}")
        print(f"Relative Residual: {residual.item():.6e}")
        
        if residual.item() < 1e-3:
            print("✅ Solver Converged!")
        else:
            print("⚠️ Solver did not converge fully.")

    # --- 5. Comparison ---
    print("\nRunning PyTorch Dense Cholesky Solve (CPU) for comparison...")
    t0 = time.time()
    try:
        x_torch = torch.linalg.solve(A_dense, b)
        t1 = time.time()
        print(f"PyTorch Dense Solve Time: {(t1-t0)*1000:.2f} ms")
    except Exception as e:
        print(f"PyTorch Solve failed (Matrix might be singular): {e}")

if __name__ == "__main__":
    os.system("pip install . --no-build-isolation") 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--density", type=float, default=0.05)
    args = parser.parse_args()
    
    benchmark(args.rows, args.density)