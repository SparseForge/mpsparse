import torch
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import os

# IMPORT YOUR COMPILED EXTENSION HERE
# import csr_extension as spmv 
# (Assuming the module name defined in setup.py is 'csr_extension')
import spmv

# Constants for numerical stability
EPSILON = 1e-8

def generate_convergent_problem(rows, cols, density=0.01):
    """
    Generates a sparse matrix A that is:
    1. Symmetric Positive Definite (SPD).
    2. Diagonally Dominant.
    3. Scaled such that eigenvalues are approx ~1000.
    """
    print(f"Generating {rows}x{cols} sparse matrix with density {density}...")
    
    # 1. Generate random sparse matrix
    A_rand = sp.random(rows, cols, density=density, format='coo', dtype=np.float32)
    
    # 2. Make it symmetric (A + A.T)
    A_sym = (A_rand + A_rand.T) / 2
    
    # 3. Enforce Diagonal Dominance & Scaling
    # Richardson with w=0.001 is stable if lambda < 2000.
    # We sum absolute values of rows to get the "off-diagonal mass"
    row_sums = np.array(np.abs(A_sym.sum(axis=1))).flatten()
    
    # SAFETY: If row_sums are too high, lambda > 2000 and solver explodes (NaNs).
    # We clip the off-diagonal mass to ensure stability if random gen creates dense rows.
    # Max safe off-diagonal sum approx 450 to keep max lambda < ~1900
    # (Lambda_max <= diag + row_sum = (row_sum + 1000) + row_sum)
    if np.max(row_sums) > 450:
        scale_factor = 450.0 / np.max(row_sums)
        A_sym = A_sym * scale_factor
        row_sums = row_sums * scale_factor

    # Set diagonal = row_sum + padding to ensure SPD.
    # We center the diagonal around 1000.0 so 0.001 * 1000 = 1.0 step size.
    diag_padding = 1000.0 
    A_sym.setdiag(row_sums + diag_padding)
    
    # 4. Create Ground Truth x and calc b = Ax
    x_true = np.random.randn(cols).astype(np.float32)
    b = A_sym.dot(x_true)
    
    return A_sym.tocoo(), x_true, b

def run_benchmark():
    # Setup
    N = 4096 # Matrix size
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate data
    A_coo, x_true_np, b_np = generate_convergent_problem(N, N)
    
    # Convert to Tensors
    rows_t = torch.from_numpy(A_coo.row).int()
    cols_t = torch.from_numpy(A_coo.col).int()
    vals_t = torch.from_numpy(A_coo.data).float()
    
    # Pack keys for the custom constructor (row << 32 | col)
    keys_t = (rows_t.long() << 32) | cols_t.long()
    
    # Prepare Inputs
    b_t = torch.from_numpy(b_np).float()
    x_init_t = torch.zeros(N, dtype=torch.float32) # Start guess at 0
    
    # Dummy tensors for CSR output
    out_row_ptr = torch.zeros(N + 1, dtype=torch.int32)
    out_col_ind = torch.zeros(len(vals_t), dtype=torch.int32)
    out_vals = torch.zeros(len(vals_t), dtype=torch.float32)

    print("\n--- Initializing Metal CSR Tensor ---")
    start_init = time.time()
    
    # Initialize implementation
    try:
        solver = spmv.csr_tensor(
            keys_t, 
            vals_t, 
            out_row_ptr, 
            out_col_ind, 
            out_vals, 
            N, N
        )
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"Initialization (COO->CSR + Alloc) time: {time.time() - start_init:.4f}s")

    # --- Run Iterative Solve ---
    print("\n--- Running Iterative Solve (1000 iters) ---")
    
    start_solve = time.time()
    
    # This runs the 1000 iterations in C++
    solver.iter_solve(b_t, x_init_t)
    
    end_solve = time.time()
    print(f"Solve Time: {end_solve - start_solve:.4f}s")
    
    # --- Validation ---
    x_final_np = x_init_t.numpy()
    
    # CHECK: Detect NaNs from solver explosion immediately
    if np.isnan(x_final_np).any() or np.isinf(x_final_np).any():
        print("\n⚠️  WARNING: Solver output contains NaNs or Infs.")
        print("   This usually means the matrix eigenvalues > 2000 (instability).")
        rel_error = float('inf')
    else:
        # Calculate Error
        # Relative Error: ||x_true - x_est|| / ||x_true||
        # ADDED EPSILON to denominator to prevent div-by-zero
        error_norm = np.linalg.norm(x_true_np - x_final_np)
        true_norm = np.linalg.norm(x_true_np)
        rel_error = error_norm / (true_norm + EPSILON)
    
    print(f"\n--- Results ---")
    print(f"Relative Error: {rel_error:.6f}")
    
    if rel_error < 1e-2:
        print("✅ SUCCESS: Solver converged significantly.")
    else:
        print("⚠️  WARNING: Convergence poor or unstable.")

    # --- CPU Baseline Comparison ---
    print("\n--- CPU Baseline (Scipy CG) ---")
    start_cpu = time.time()
    # Scipy CG is generally robust, but we wrap in try/except just in case
    try:
        x_cpu, info = sp.linalg.cg(A_coo, b_np, maxiter=1000, rtol=1e-5)
        end_cpu = time.time()
        print(f"CPU Time: {end_cpu - start_cpu:.4f}s")
    except Exception as e:
        print(f"CPU Solver Failed: {e}")

if __name__ == "__main__":
    os.system("pip install . --no-build-isolation") 
    run_benchmark()