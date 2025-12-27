import torch
import time
import spmv # Your compiled extension

# --- 1. Data Generators (Unchanged) ---

def pack_keys(rows, cols):
    return (rows.to(torch.int64) << 32) | cols.to(torch.int64)

def generate_laplacian_2d(grid_size):
    """Physics: 2D Finite Difference Grid"""
    N = grid_size * grid_size
    print(f"   -> Generating Physics Grid ({grid_size}x{grid_size}, N={N})...")
    offsets = [0, -1, 1, -grid_size, grid_size]
    all_rows, all_cols = [], []
    idx = torch.arange(N, dtype=torch.int32)
    for k in offsets:
        r, c = idx, idx + k
        valid = (c >= 0) & (c < N)
        if abs(k) == 1: 
            valid = valid & ((r // grid_size) == (c // grid_size))
        all_rows.append(r[valid])
        all_cols.append(c[valid])
    return torch.cat(all_rows).long(), torch.cat(all_cols).long(), N

def generate_block_diagonal(num_blocks, block_size, density=0.2):
    """Control: Block Diagonal"""
    total_rows = num_blocks * block_size
    print(f"   -> Generating Block Diagonal ({num_blocks} blocks, N={total_rows})...")
    all_rows, all_cols = [], []
    for i in range(num_blocks):
        nnz = int(block_size * block_size * density)
        r = torch.randint(0, block_size, (nnz,), dtype=torch.int64)
        c = torch.randint(0, block_size, (nnz,), dtype=torch.int64)
        all_rows.append(r + i * block_size)
        all_cols.append(c + i * block_size)
    
    packed = (torch.cat(all_rows) << 32) | torch.cat(all_cols)
    packed = torch.unique(packed)
    return (packed >> 32), (packed & 0xFFFFFFFF), total_rows

def generate_pruned_layer(rows, cols, density=0.1):
    """ML: Pruned Weight Matrix"""
    print(f"   -> Generating ML Pruned Layer ({rows}x{cols}, Density={density})...")
    nnz = int(rows * cols * density)
    oversample = int(nnz * 1.1)
    r = torch.randint(0, rows, (oversample,), dtype=torch.int64)
    c = torch.randint(0, cols, (oversample,), dtype=torch.int64)
    packed = (r << 32) | c
    packed = torch.unique(packed)[:nnz]
    return (packed >> 32), (packed & 0xFFFFFFFF), rows

# --- 2. Throughput Calculator ---

def get_spmv_bandwidth_gb(rows, cols, nnz):
    # Calculate total memory traffic for one SpMV operation
    # Read CSR: (NNZ * 4 bytes val) + (NNZ * 4 bytes col) + (Rows * 4 bytes ptr)
    # Read X:   (Cols * 4 bytes)
    # Write Y:  (Rows * 4 bytes)
    total_bytes = (nnz * 8) + (rows * 4) + (cols * 4) + (rows * 4)
    return total_bytes / 1e9

def get_conversion_bandwidth_gb(nnz, rows):
    # Approximate traffic for COO -> CSR
    # Read COO: 2 * NNZ * 4 bytes (or 1 * NNZ * 8 bytes packed)
    # Write CSR: NNZ * 4 + NNZ * 4 + Rows * 4
    total_bytes = (nnz * 8) + (nnz * 8) + (rows * 4)
    return total_bytes / 1e9

# --- 3. Benchmark Engine ---

def run_benchmark(dataset_name):
    print(f"\n================ {dataset_name} ================")
    
    # --- A. DATA GENERATION ---
    if dataset_name == "Physics":
        r, c, N = generate_laplacian_2d(grid_size=2048) 
        M = N
    elif dataset_name == "Block":
        r, c, N = generate_block_diagonal(num_blocks=1000, block_size=100) 
        M = N
    elif dataset_name == "ML":
        N, M = 25192, 25192 
        r, c, _ = generate_pruned_layer(N, M, density=0.1) 

    nnz = r.size(0)
    values = torch.rand(nnz, dtype=torch.float32)
    packed_keys = pack_keys(r, c)
    
    print(f"   Matrix: {N}x{M} | NNZ: {nnz}")
    
    x = torch.randn(M, dtype=torch.float32)
    y_custom = torch.zeros(N, dtype=torch.float32)
    
    # Dummy outputs for your constructor
    dummy_row_ptr = torch.zeros(N + 1, dtype=torch.int32)
    dummy_col_ind = torch.zeros(nnz, dtype=torch.int32)
    dummy_vals = torch.zeros(nnz, dtype=torch.float32)

    # --- B. WARMUP (CRITICAL STEP) ---
    print("\n   [Warmup Phase]")
    print("   Running one full pass to page-in memory and cache kernels...")
    warmup_obj = spmv.csr_tensor(
        packed_keys, values, dummy_row_ptr, dummy_col_ind, dummy_vals, N, M
    )
    warmup_obj.mv(x, y_custom)
    del warmup_obj 
    
    # --- C. CUSTOM METAL BENCHMARK ---
    print("\n   [1] Metal Implementation")
    
    # 1. Measure Setup (COO -> CSR Conversion)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_start = time.time()
    
    csr_obj = spmv.csr_tensor(
        packed_keys, values, 
        dummy_row_ptr, dummy_col_ind, dummy_vals, 
        N, M
    )
    
    t_setup_metal = (time.time() - t_start)
    
    # 2. Measure Compute (SpMV)
    iters = 100
    t_start = time.time()
    for _ in range(iters):
        csr_obj.mv(x, y_custom)
    t_compute_metal = (time.time() - t_start) / iters

    # --- D. PYTORCH CPU BASELINE ---
    print("   [2] PyTorch CPU Implementation")
    
    # 1. Measure Setup (COO -> CSR Conversion)
    # PyTorch sparse tensors are complex, so we measure the creation + coalescing
    t_start = time.time()
    torch_coo = torch.sparse_coo_tensor(
        torch.stack([r, c]), values, (N, M), dtype=torch.float32
    )
    torch_csr = torch_coo.to_sparse_csr()
    t_setup_torch = (time.time() - t_start)
    
    # 2. Measure Compute (SpMV)
    x_matrix = x.unsqueeze(1) 
    # Warmup torch
    for _ in range(5): res = torch.mm(torch_csr, x_matrix)
        
    t_start = time.time()
    for _ in range(iters):
        res = torch.mm(torch_csr, x_matrix)
    t_compute_torch = (time.time() - t_start) / iters

    # --- E. REPORTING & THROUGHPUT ---
    print("\n   --- Results ---")
    
    gb_conversion = get_conversion_bandwidth_gb(nnz, N)
    gb_spmv = get_spmv_bandwidth_gb(N, M, nnz)

    print(f"   {'Metric':<20} | {'Metal (Yours)':<15} | {'PyTorch CPU':<15} | {'Speedup':<10}")
    print(f"   {'-'*70}")
    
    # Setup Metrics
    print(f"   {'Setup Time':<20} | {t_setup_metal*1000:8.2f} ms    | {t_setup_torch*1000:8.2f} ms    | {t_setup_torch/t_setup_metal:5.2f}x")
    print(f"   {'Setup Throughput':<20} | {gb_conversion/t_setup_metal:8.2f} GB/s  | {gb_conversion/t_setup_torch:8.2f} GB/s  |")
    print(f"   {'-'*70}")
    
    # Compute Metrics
    print(f"   {'Compute Time':<20} | {t_compute_metal*1000:8.4f} ms    | {t_compute_torch*1000:8.4f} ms    | {t_compute_torch/t_compute_metal:5.2f}x")
    print(f"   {'Compute Throughput':<20} | {gb_spmv/t_compute_metal:8.2f} GB/s  | {gb_spmv/t_compute_torch:8.2f} GB/s  |")
    
    # Correctness Check
    ref = res.flatten()
    out = y_custom.flatten()
    if torch.allclose(ref, out, atol=1e-3, rtol=1e-3):
        print("\n   ✅ Verification Passed")
    else:
        print("\n   ❌ Verification Failed (Max Diff: {:.4f})".format((ref - out).abs().max()))

if __name__ == "__main__":
    import os
    os.system("pip install . --no-build-isolation") 
    
    run_benchmark("Block")
    run_benchmark("Physics")
    run_benchmark("ML")