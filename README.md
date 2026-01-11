# MPSparse

An implementation of sparse matrix operations using Apple's Metal Performance Shaders. The goal is to build a library which can support QR and LU factorization of sparse matrices 

TODO:
* Test COO to CSR converter, drag race against CUDA impl
* Test SpMV kernel, again vs. CUDA impl
* Implement sparse QR in CUDA (reference)
* Discuss iterative solvers more
* Plug in COO to CSR to torch
