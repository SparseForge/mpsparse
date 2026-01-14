# MPSparse

An implementation of sparse matrix operations using Apple's Metal Performance Shaders. The goal is to build a library which can support QR and LU factorization of sparse matrices 

## Currently Implemented:
* COO to CSR converters
* CGD

TODO:
* Test COO to CSR converter, drag race against CUDA impl
* Test SpMV kernel, again vs. CUDA impl
* Implement sparse QR in CUDA (reference)
* Discuss iterative solvers more
* Plug in COO to CSR to torch
* BiCGSTAB

# CGD

This repository implements a high-performance Conjugate Gradient Descent (CGD) solver for sparse linear systems, accelerated by Apple Metal on macOS devices.

It features a custom PyTorch extension that performs Sparse Matrix-Vector multiplication (SpMV) and iterative solver steps entirely on the GPU. By leveraging fused kernels and Metal's SIMD-group functions, this implementation minimizes memory bandwidth usage and CPU-GPU synchronization overhead compared to standard solvers.

## Usage
1. Data Preparation (Key Packing)

The solver accepts input in Coordinate (COO) format. To efficiently sort and convert this to CSR format on the GPU, the row and column indices must be packed into a single 64-bit integer key.

The packing logic places the row index in the high 32 bits and the column index in the low 32 bits.

```python
import torch

def pack_keys(row_indices, col_indices):
    """
    Packs row and column indices into a single uint64 key.
    High 32 bits: Row Index
    Low 32 bits: Col Index
    """
    row_indices = row_indices.to(torch.int64)
    col_indices = col_indices.to(torch.int64)
    return (row_indices << 32) | col_indices
```

2. Running the Solver

Below is an example of initializing the custom tensor and solving $Ax=b$.

```python
import torch
import spmv 

# Define problem size
num_rows = 1024
num_cols = 1024

# Create a sample diagonal matrix (Indices and Values)
rows = torch.arange(num_rows)
cols = torch.arange(num_cols)
values = torch.ones(num_rows, dtype=torch.float32)

# Create RHS vector 'b' and initial guess 'x'
b = torch.randn(num_rows, dtype=torch.float32)
x_solution = torch.zeros(num_cols, dtype=torch.float32)

# Pack keys for the internal sorter
keys = pack_keys(rows, cols)

# Initialize the Metal CSR Tensor
# The constructor handles the COO -> CSR conversion on the GPU
tensor = spmv.csr_tensor(
    keys,           
    values,         
    torch.zeros(num_rows + 1, dtype=torch.int32),     # Buffer for row_ptr
    torch.zeros(keys.shape[0], dtype=torch.int32),    # Buffer for col_ind
    torch.zeros(keys.shape[0], dtype=torch.float32),  # Buffer for sorted values
    num_rows,
    num_cols
)

# Run the Iterative Solver
# The result is written in-place to x_solution
tensor.iter_solve(b, x_solution)

print(f"Residual: {torch.norm(values * x_solution - b)}")
```

## Benchmarks

We benchmarked the solver against standard CPU implementations using two distinct matrix topologies:

  1. ML Block-Sparse: Structured sparsity common in pruned machine learning models.

  2. Physics Stencil: 5-point Laplacian stencil common in finite difference simulations.

Performance Results

![ML Matrices](graphs/benchmark_ml_results.png)
![Physics Matrices](graphs/benchmark_physics_results.png)

# BiCGSTAB

# Future Steps 

Our COO to CSR conversion occurs on the GPU but takes longer than anticipated, so a major goal is to optimize those kernels. For CGD, our goal is to implement Blocked CSR (BCSR) to speed up operations. We also hope to implement adaptive kernels; this will likely take longer and may not happen for a while.

