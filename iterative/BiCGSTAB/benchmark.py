import torch
import spmv  # Your custom extension
import time
import math
import os
import sys

# --- Auto-Install Dependencies for Plotting ---
def install_plotting_deps():
    print("Checking and installing plotting dependencies...")
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        os.system(f"{sys.executable} -m pip install pandas matplotlib seaborn --no-build-isolation")
        print("Dependencies installed.")

install_plotting_deps()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure reproducibility
torch.manual_seed(42)

# --- 1. Matrix Generators ---

def generate_ml_block_sparse(rows, cols, block_size=32, density=0.05):
    """
    Generates a Block-Sparse Diagonally Dominant Matrix (Nonsymmetric).
    Used to test ML-like sparsity patterns.
    """
    n_blocks_row = (rows + block_size - 1) // block_size
    n_blocks_col = (cols + block_size - 1) // block_size
    
    # 1. Block Mask
    block_mask = torch.rand(n_blocks_row, n_blocks_col) < density
    
    # Upscale mask to full size
    mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    mask = mask[:rows, :cols]
    
    # 2. Random Values
    A_dense = torch.randn(rows, cols) * mask.float()
    
    # 3. Make Diagonally Dominant (for BiCGSTAB stability)
    # A_ii = sum(|A_ij|) + epsilon
    row_sums = torch.sum(torch.abs(A_dense), dim=1)
    A_dense.as_strided([rows], [cols + 1]).copy_(row_sums + 1.0)
    
    # 4. Extract COO format for Custom Extension
    A_sparse = A_dense.to_sparse_coo()
    
    return A_sparse, A_dense

def generate_physics_stencil(grid_n):
    """
    Generates a 5-point 2D Laplacian Stencil (Finite Difference).
    Size N = grid_n * grid_n.
    Typical of Physics/CFD simulations.
    """
    N = grid_n * grid_n
    
    # Indices
    indices = []
    values = []
    
    # We build the stencil explicitly
    # Center: 4.0, Neighbors: -1.0
    
    rows = torch.arange(N)
    cols = torch.arange(N)
    
    # Main Diagonal
    indices.append(torch.stack([rows, cols]))
    values.append(torch.full((N,), 4.0))
    
    # Off-diagonals (Left/Right)
    # Right Neighbor (i, j+1) exists if (i % grid_n) != (grid_n - 1)
    mask_right = (rows % grid_n) != (grid_n - 1)
    r_right = rows[mask_right]
    c_right = rows[mask_right] + 1
    indices.append(torch.stack([r_right, c_right]))
    values.append(torch.full((r_right.shape[0],), -1.0))
    
    indices.append(torch.stack([c_right, r_right])) # Symmetric
    values.append(torch.full((r_right.shape[0],), -1.0))

    # Off-diagonals (Up/Down)
    # Down Neighbor (i+1, j) exists if i < N - grid_n
    mask_down = rows < (N - grid_n)
    r_down = rows[mask_down]
    c_down = rows[mask_down] + grid_n
    indices.append(torch.stack([r_down, c_down]))
    values.append(torch.full((r_down.shape[0],), -1.0))
    
    indices.append(torch.stack([c_down, r_down])) # Symmetric
    values.append(torch.full((r_down.shape[0],), -1.0))
    
    all_indices = torch.cat(indices, dim=1)
    all_values = torch.cat(values)
    
    A_sparse = torch.sparse_coo_tensor(all_indices, all_values, (N, N)).coalesce()
    
    return A_sparse

# --- 2. Utilities ---

def pack_keys(row_indices, col_indices):
    """Packs two 32-bit indices into one 64-bit key for the Sort."""
    row_indices = row_indices.to(torch.int64)
    col_indices = col_indices.to(torch.int64)
    return (row_indices << 32) | col_indices

def bicgstab_pytorch_dense(A, b, max_iter=1000, tol=1e-5):
    """
    Pure PyTorch implementation of BiCGSTAB (Dense) for baseline.
    """
    x = torch.zeros_like(b)
    r = b - torch.matmul(A, x)
    r0_hat = r.clone()
    
    rho_old = 1.0
    alpha = 1.0
    omega = 1.0
    v = torch.zeros_like(b)
    p = torch.zeros_like(b)
    
    for i in range(max_iter):
        rho = torch.dot(r0_hat, r)
        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        
        v = torch.matmul(A, p)
        sigma = torch.dot(r0_hat, v)
        alpha = rho / sigma
        
        s = r - alpha * v
        if torch.norm(s) < tol:
            x = x + alpha * p
            break
            
        t = torch.matmul(A, s)
        
        omega = torch.dot(t, s) / torch.dot(t, t)
        x = x + alpha * p + omega * s
        r = s - omega * t
        
        if torch.norm(r) < tol:
            break
            
        rho_old = rho
        
    return x

# --- 3. Benchmarking Logic ---

def benchmark_single_run(mode, size, density=None):
    print(f"--- Running: {mode.upper()} | Size: {size}x{size} ---")
    
    # A. Generate Data
    if mode == "ml":
        A_sparse, A_dense = generate_ml_block_sparse(size, size, density=density)
    else: # physics
        grid_n = int(math.sqrt(size))
        size = grid_n * grid_n # Adjust size to be exact square
        A_sparse = generate_physics_stencil(grid_n)
        A_dense = A_sparse.to_dense()

    rows, cols = size, size
    indices = A_sparse.indices()
    values = A_sparse.values().to(torch.float32)
    nnz = values.shape[0]
    
    # B. Setup Inputs
    x_target = torch.randn(cols)
    b = torch.matmul(A_dense, x_target)
    
    # C. Initialize Custom Extension
    # Pre-allocate buffers required by the C++ constructor
    keys = pack_keys(indices[0], indices[1])
    row_ptr = torch.zeros(rows + 1, dtype=torch.int32)
    col_ind = torch.zeros(nnz, dtype=torch.int32)
    out_vals = torch.zeros(nnz, dtype=torch.float32)
    
    # Init Timer
    t0 = time.time()
    # Constructor does the Heavy Sorting & Compression on GPU
    solver = spmv.csr_tensor(keys, values, row_ptr, col_ind, out_vals, rows, cols)
    t_setup = time.time() - t0
    
    # D. Benchmark Custom Solve
    x_custom = torch.zeros(rows)
    
    # Warmup
    solver.bicgstab(b, x_custom) 
    
    # Timed Run
    iterations = 5
    t_start = time.time()
    for _ in range(iterations):
        x_custom.zero_() # Reset X
        solver.bicgstab(b, x_custom)
    torch.cuda.synchronize() if torch.cuda.is_available() else None # Placeholder for Sync
    t_custom_avg = (time.time() - t_start) / iterations

    # E. Benchmark PyTorch Baseline
    # We use the Python Loop BiCGSTAB to be algorithmically fair
    t_start = time.time()
    bicgstab_pytorch_dense(A_dense, b) # Just one run as it's slow
    t_torch = time.time() - t_start
    
    # Verify Accuracy
    diff = torch.norm(b - torch.matmul(A_dense, x_custom)) / torch.norm(b)
    print(f"   -> Custom Residual: {diff:.6f}")
    
    return {
        "Mode": mode,
        "Size": size,
        "NNZ": nnz,
        "Setup Time (s)": t_setup,
        "Custom Time (s)": t_custom_avg,
        "PyTorch Time (s)": t_torch,
        "Speedup": t_torch / t_custom_avg
    }

# --- 4. Main Execution ---

def main():
    results = []
    
    print("========================================")
    print("Starting Benchmark Suite")
    print("========================================\n")
    
    # 1. Physics Benchmarks (Laplacian Stencil)
    grid_sizes = [32, 64, 100, 128] 
    for n in grid_sizes:
        try:
            res = benchmark_single_run("physics", n*n)
            results.append(res)
        except Exception as e:
            print(f"Error on Physics {n}: {e}")

    # 2. ML Benchmarks (Block Sparse)
    sizes = [1024, 4096, 8192]
    density = 0.01
    for s in sizes:
        try:
            res = benchmark_single_run("ml", s, density=density)
            results.append(res)
        except Exception as e:
            print(f"Error on ML {s}: {e}")

    # --- SAVE RESULTS ---
    df = pd.DataFrame(results)
    if df.empty:
        print("No results collected.")
        return

    # SAVE TO CSV HERE
    csv_filename = "benchmark_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n[Success] Raw data saved to {csv_filename}")

    print("\n=== Results Summary ===")
    print(df[["Mode", "Size", "Custom Time (s)", "PyTorch Time (s)", "Speedup"]])
    
    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Speedup vs Size
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Size", y="Speedup", hue="Mode", palette="viridis")
        plt.title("GPU BiCGSTAB Speedup vs PyTorch Loop")
        plt.ylabel("Speedup Factor (x)")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig("benchmark_speedup.png")
        print("Saved benchmark_speedup.png")
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    # Plot 2: Absolute Time
    try:
        plt.figure(figsize=(10, 6))
        df_melt = df.melt(id_vars=["Size", "Mode"], 
                          value_vars=["Custom Time (s)", "PyTorch Time (s)"], 
                          var_name="Impl", value_name="Time")
        sns.lineplot(data=df_melt, x="Size", y="Time", hue="Impl", style="Mode", markers=True)
        plt.yscale("log")
        plt.title("Execution Time: Custom GPU vs PyTorch Baseline")
        plt.ylabel("Time (seconds) [Log Scale]")
        plt.tight_layout()
        plt.savefig("benchmark_time.png")
        print("Saved benchmark_time.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    os.system("pip install . --no-build-isolation")
    main()