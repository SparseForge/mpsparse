#note to execute 
#ln -s ../coo_csr/coo_csr/coo_csr/metal-cpp ./metal-cpp
#or wherever u have that file

import torch
import numpy as np
import scipy.sparse
import time
import spmv

# --- Configuration ---
# Sizes: (Rows, Cols, Density)
SIZES = [
    (4096, 4096, 0.01),    
    (10000, 10000, 0.005), 
    (50000, 50000, 0.005),
    (100000, 100000, 0.005),
]
NUM_ITER = 50
NUM_WARMUP = 10

def generate_data(rows, cols, density):
    nnz = int(rows * cols * density)
    
    # Generate random COO
    row_ind = torch.randint(0, rows, (nnz,), dtype=torch.int32)
    col_ind = torch.randint(0, cols, (nnz,), dtype=torch.int32)
    values = torch.randn(nnz, dtype=torch.float32)
    
    # Create Scipy baseline
    scipy_mat = scipy.sparse.coo_matrix(
        (values.numpy(), (row_ind.numpy(), col_ind.numpy())), 
        shape=(rows, cols)
    ).tocsr()

    # Pack keys for Metal (row << 32 | col)
    packed_keys = (row_ind.to(torch.int64) << 32) | col_ind.to(torch.int64)
    
    # Pre-allocate output buffers
    out_row_ptr = torch.zeros(rows + 1, dtype=torch.int32)
    out_col_ind = torch.zeros(nnz, dtype=torch.int32)
    out_vals = torch.zeros(nnz, dtype=torch.float32)

    return (packed_keys, values, out_row_ptr, out_col_ind, out_vals), scipy_mat

def benchmark():
    # Header
    print(f"{'Size':<15} | {'Setup':<8} | {'CPU (ms)':<9} | {'GPU (ms)':<9} | {'CPU GB/s':<9} | {'GPU GB/s':<9} | {'Speedup':<8} | {'Status'}")
    print("-" * 115)

    for rows, cols, density in SIZES:
        metal_inputs, scipy_mat = generate_data(rows, cols, density)
        packed_keys, values, out_rp, out_ci, out_v = metal_inputs
        
        nnz = int(rows * cols * density)

        # --- Calculate Theoretical Traffic ---
        # 1. Matrix Reads: (vals + col_ind) * nnz + row_ptr * (rows+1)
        # 2. Vector Read: x * cols
        # 3. Vector Write: b * rows
        # All are 4 bytes (float32 or int32)
        total_bytes = 4 * (2 * nnz + (rows + 1) + cols + rows)
        total_gb = total_bytes / 1e9

        x = torch.randn(cols, dtype=torch.float32)
        b = torch.zeros(rows, dtype=torch.float32)
        
        # 1. Setup (Convert COO -> CSR on GPU)
        t0 = time.time()
        custom_csr = spmv.csr_tensor(
            packed_keys, values, out_rp, out_ci, out_v, rows, cols
        )
        if torch.cuda.is_available(): torch.cuda.synchronize()
        setup_ms = (time.time() - t0) * 1000

        # 2. Scipy Benchmark
        x_np = x.numpy()
        for _ in range(5): scipy_mat.dot(x_np) # Warmup
        
        t0 = time.time()
        for _ in range(NUM_ITER):
            res = scipy_mat.dot(x_np)
        cpu_ms = ((time.time() - t0) / NUM_ITER) * 1000

        # 3. Metal Benchmark
        for _ in range(NUM_WARMUP): custom_csr.mv(x, b) # Warmup
        
        t0 = time.time()
        for _ in range(NUM_ITER):
            custom_csr.mv(x, b)
        gpu_ms = ((time.time() - t0) / NUM_ITER) * 1000

        # 4. Bandwidth Calculations
        # GB/s = Total GB / (Time in seconds)
        cpu_gbps = total_gb / (cpu_ms / 1000.0)
        gpu_gbps = total_gb / (gpu_ms / 1000.0)

        # 5. Check correctness
        ref = scipy_mat.dot(x_np)
        diff = np.linalg.norm(ref - b.numpy()) / (np.linalg.norm(ref) + 1e-6)
        status = "PASS" if diff < 1e-4 else f"FAIL ({diff:.1e})"
        
        # Print Result
        print(f"{rows}x{cols:<9} | {setup_ms:<8.1f} | {cpu_ms:<9.3f} | {gpu_ms:<9.3f} | {cpu_gbps:<9.2f} | {gpu_gbps:<9.2f} | {cpu_ms/gpu_ms:<8.2f} | {status}")

if __name__ == "__main__":
    import os
    os.system("pip install . --no-build-isolation")
    benchmark()
