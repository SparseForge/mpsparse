import torch
import coo_csr  
import time

# --- Helpers ---
def pack_keys(rows, cols):
    return (rows.to(torch.int64) << 32) | cols.to(torch.int64)

def unpack_keys(packed):
    rows = (packed >> 32).to(torch.int32)
    cols = (packed & 0xFFFFFFFF).to(torch.int32)
    return rows, cols

# --- Generator 1: Banded Matrix (Physics/Engineering style) ---
# Creates a Diagonally Dominant matrix. Fast and realistic.
def generate_banded_matrix(num_rows, num_cols, num_diagonals=7):
    print(f"Generating Banded Matrix (Offsets: {num_diagonals})...")
    
    offsets = [0]
    for i in range(1, (num_diagonals // 2) + 1):
        offsets.append(i)
        offsets.append(-i)
        offsets.append(i * 10) 
        offsets.append(-i * 10)
        
    all_rows = []
    all_cols = []
    
    for k in offsets:
        if k >= 0:
            count = min(num_rows, num_cols - k)
            if count <= 0: continue
            r = torch.arange(count, dtype=torch.int32)
            c = r + k
        else:
            count = min(num_rows + k, num_cols)
            if count <= 0: continue
            c = torch.arange(count, dtype=torch.int32)
            r = c - k
            
        all_rows.append(r)
        all_cols.append(c)
        
    row_indices = torch.cat(all_rows)
    col_indices = torch.cat(all_cols)
    

    packed = (row_indices.to(torch.int64) << 32) | col_indices.to(torch.int64)
    packed = torch.unique(packed)
    
    row_indices = (packed >> 32).to(torch.int32)
    col_indices = (packed & 0xFFFFFFFF).to(torch.int32)
    
    perm = torch.randperm(row_indices.size(0))
    row_indices = row_indices[perm]
    col_indices = col_indices[perm]
    
    values = torch.rand(row_indices.size(0), dtype=torch.float32)
    return row_indices, col_indices, values

# --- Generator 2: Fast Random (Vectorized) ---
# Uses torch.unique to handle duplicates instantly.
def generate_random_fast(num_rows, num_cols, sparsity=0.01):
    target_nnz = int(num_rows * num_cols * sparsity)
    print(f"Generating Random Matrix ({target_nnz} elements)...")
    
    oversample = int(target_nnz * 1.1)
    r = torch.randint(0, num_rows, (oversample,), dtype=torch.int32)
    c = torch.randint(0, num_cols, (oversample,), dtype=torch.int32)
    
    keys = (r.to(torch.int64) << 32) | c.to(torch.int64)
    unique_keys = torch.unique(keys)
    
    if unique_keys.size(0) > target_nnz:
        unique_keys = unique_keys[:target_nnz]
        
    row_indices, col_indices = unpack_keys(unique_keys)
    values = torch.rand(row_indices.size(0), dtype=torch.float32)
    
    return row_indices, col_indices, values

# --- Main Benchmark ---
def run_benchmark(num_rows, num_cols, mode='banded'):
    print(f"\n--- BENCHMARK: {num_rows}x{num_cols} | Mode: {mode} ---")
    
    # 1. GENERATION
    t0 = time.time()
    if mode == 'banded':
        # num_diagonals=27 approximates a 3D finite element grid
        row_indices, col_indices, values = generate_banded_matrix(num_rows, num_cols, num_diagonals=27)
    else:
        row_indices, col_indices, values = generate_random_fast(num_rows, num_cols, sparsity=0.01)
    
    nnz = row_indices.size(0)
    print(f"Data Gen Time: {(time.time()-t0)*1000:.2f} ms | NNZ: {nnz}")

    # 2. PREPARE BUFFERS
    packed_keys = pack_keys(row_indices, col_indices)
    
    # Output tensors (Standard Interface)
    out_row_ptr = torch.zeros(num_rows + 1, dtype=torch.int32)
    out_col_ind = torch.zeros(nnz, dtype=torch.int32)
    out_vals = torch.zeros(nnz, dtype=torch.float32)

    # 3. WARMUP
    # Standard call: keys, vals, row_ptr, col_ind, out_vals, num_rows
    coo_csr.coo_csr(packed_keys, values, out_row_ptr, out_col_ind, out_vals, num_rows, num_cols)

    # 4. RUN METAL BENCHMARK
    start_time = time.time()
    iters = 20
    for _ in range(iters):
        coo_csr.coo_csr(packed_keys, values, out_row_ptr, out_col_ind, out_vals, num_rows, num_cols)
    
    avg_metal_time = (time.time() - start_time) / iters * 1000
    print(f"Custom Metal Sort: {avg_metal_time:.3f} ms")

    # 5. PYTORCH BASELINE
    coo_tensor = torch.sparse_coo_tensor(
        torch.stack([row_indices.long(), col_indices.long()]), 
        values, 
        (num_rows, num_cols)
    )
    
    start_time = time.time()
    for _ in range(iters):
        csr = coo_tensor.to_sparse_csr()
    avg_torch_time = (time.time() - start_time) / iters * 1000
    print(f"PyTorch CPU Sort:  {avg_torch_time:.3f} ms")
    
    # 6. VERIFICATION
    # Compare against Ground Truth
    ref_csr = coo_tensor.to_sparse_csr()
    
    # Check Row Pointers
    if torch.equal(out_row_ptr.long(), ref_csr.crow_indices()):
        print("✅ Row Pointers Match")
    else:
        print("❌ Row Pointers Mismatch")
        # Print first few errors
        diff = torch.where(out_row_ptr.long() != ref_csr.crow_indices())[0]
        print(f"   First mismatch at row {diff[0].item()}")
        print(f"   Expected: {ref_csr.crow_indices()[diff[0]]}")
        print(f"   Got:      {out_row_ptr[diff[0]]}")

    # Check Column Indices
    if torch.equal(out_col_ind.long(), ref_csr.col_indices()):
        print("✅ Column Indices Match")
    else:
        print("❌ Column Indices Mismatch")
        
    # Check Values
    if torch.allclose(out_vals, ref_csr.values()):
        print("✅ Values Match")
    else:
        print("❌ Values Mismatch")

if __name__ == "__main__":
    import os
    os.system("pip install . --no-build-isolation")

    # Small test for correctness
    run_benchmark(num_rows=10000, num_cols=10000, mode='banded')
    
    # Large test for performance
    run_benchmark(num_rows=1100000, num_cols=1100000, mode='banded')