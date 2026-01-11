#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t status = call; \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            fprintf(stderr, "CUSPARSE Error at line %d\n", __LINE__); \
            exit(1); \
        } \
    } while (0)

// -------------------------------------------------------------------------
// Kernel: 解包 64位 Key -> 32位 Row 和 Col
// Metal 代码中 keys 是 ulong，高32位是 row，低32位是 col
// -------------------------------------------------------------------------
__global__ void unpack_keys_kernel(const uint64_t* packed_keys, int* rows, int* cols, int nnz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        uint64_t key = packed_keys[idx];
        rows[idx] = (int)(key >> 32);
        cols[idx] = (int)(key & 0xFFFFFFFF);
    }
}

// -------------------------------------------------------------------------
// Kernel: 加权加法 (Weighted Add)
// out = alpha * input1 + beta * input2
// 对应 Metal 的 pso_wadd
// -------------------------------------------------------------------------
__global__ void weighted_add_kernel(const float* input1, const float* input2, float* out, 
                                    float alpha, float beta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = alpha * input1[idx] + beta * input2[idx];
    }
}

class csr_tensor {
public:
    cusparseHandle_t cusparse_handle;
    cublasHandle_t cublas_handle;
    
    cusparseSpMatDescr_t matA;
    void* dBuffer_spmv = nullptr;
    size_t bufferSize_spmv = 0;

    int* d_csr_row_ptr = nullptr;
    int* d_csr_col_ind = nullptr;
    float* d_csr_vals = nullptr;
    
    // 辅助向量缓冲区
    float* d_buf_x = nullptr; // 用于 mv 计算的 x
    float* d_buf_b = nullptr; // 用于 mv 计算的 b (结果)
    float* d_scratch1 = nullptr;
    float* d_scratch2 = nullptr;

    uint32_t num_rows;
    uint32_t num_cols;
    uint32_t nnz;

    csr_tensor(
        torch::Tensor keys,       // 打包密钥
        torch::Tensor values, 
        torch::Tensor row_ptr,
        torch::Tensor col_ind,
        torch::Tensor out_vals,
        int num_rows,
        int num_cols
    ) {
        CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
        cublasCreate(&cublas_handle);

        this->num_rows = num_rows;
        this->num_cols = num_cols;
        this->nnz = keys.size(0);

        // 准备临时数据进行 COO 排序
        // 在 CUDA 中，我们解包 row/col，使用 Thrust 排序，然后用 cuSPARSE 转换。
        
        uint64_t* h_keys = (uint64_t*)keys.data_ptr<int64_t>();
        float* h_vals = (float*)values.data_ptr<float>();

        // 分配临时 COO 内存
        uint64_t* d_packed_keys;
        int* d_coo_rows;
        int* d_coo_cols;
        float* d_coo_vals;

        CUDA_CHECK(cudaMalloc(&d_packed_keys, nnz * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_coo_rows, nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_coo_cols, nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_coo_vals, nnz * sizeof(float)));

        // 拷贝输入数据到 GPU
        CUDA_CHECK(cudaMemcpy(d_packed_keys, h_keys, nnz * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coo_vals, h_vals, nnz * sizeof(float), cudaMemcpyHostToDevice));

        // 解包 Key (内核)
        int threads = 256;
        int blocks = (nnz + threads - 1) / threads;
        unpack_keys_kernel<<<blocks, threads>>>(d_packed_keys, d_coo_rows, d_coo_cols, nnz);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 排序 (Thrust)
        
        thrust::device_ptr<uint64_t> t_keys(d_packed_keys);
        thrust::device_ptr<float> t_vals(d_coo_vals);
        thrust::sort_by_key(t_keys, t_keys + nnz, t_vals);

        // 再次解包（现在是排序后的）
        unpack_keys_kernel<<<blocks, threads>>>(d_packed_keys, d_coo_rows, d_coo_cols, nnz);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 分配 CSR 最终内存
        CUDA_CHECK(cudaMalloc(&d_csr_row_ptr, (num_rows + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_csr_col_ind, nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_csr_vals, nnz * sizeof(float)));

        // 拷贝 Col 和 Val 到最终位置 (因为它们已经是排序好的)
        CUDA_CHECK(cudaMemcpy(d_csr_col_ind, d_coo_cols, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_csr_vals, d_coo_vals, nnz * sizeof(float), cudaMemcpyDeviceToDevice));

        // COO 转 CSR
        // 使用 cuSPARSE legacy 或 helper function
        CUSPARSE_CHECK(cusparseXcoo2csr(cusparse_handle, 
                                        d_coo_rows, 
                                        nnz, 
                                        num_rows, 
                                        d_csr_row_ptr, 
                                        CUSPARSE_INDEX_BASE_ZERO));

        // 将结果拷回 CPU
        CUDA_CHECK(cudaMemcpy(row_ptr.data_ptr<int32_t>(), d_csr_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(col_ind.data_ptr<int32_t>(), d_csr_col_ind, nnz * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_vals.data_ptr<float>(), d_csr_vals, nnz * sizeof(float), cudaMemcpyDeviceToHost));

        // 初始化 SpMV 所需的描述符
        CUSPARSE_CHECK(cusparseCreateCsr(&matA, num_rows, num_cols, nnz,
                                         d_csr_row_ptr, d_csr_col_ind, d_csr_vals,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        // 分配工作缓冲区
        CUDA_CHECK(cudaMalloc(&d_buf_x, num_cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_buf_b, num_rows * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scratch1, num_rows * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scratch2, num_rows * sizeof(float)));

        // 清理临时内存
        CUDA_CHECK(cudaFree(d_packed_keys));
        CUDA_CHECK(cudaFree(d_coo_rows));
        CUDA_CHECK(cudaFree(d_coo_cols));
        CUDA_CHECK(cudaFree(d_coo_vals));
    }

    ~csr_tensor() {
        if (matA) cusparseDestroySpMat(matA);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (cublas_handle) cublasDestroy(cublas_handle);
        
        if (dBuffer_spmv) cudaFree(dBuffer_spmv);
        
        cudaFree(d_csr_row_ptr);
        cudaFree(d_csr_col_ind);
        cudaFree(d_csr_vals);
        cudaFree(d_buf_x);
        cudaFree(d_buf_b);
        cudaFree(d_scratch1);
        cudaFree(d_scratch2);
    }

    // 内部 SpMV: b = A * x
    void mv_internal(float* x_ptr, float* b_ptr) {
        // 拷贝 x 到 GPU
        CUDA_CHECK(cudaMemcpy(d_buf_x, x_ptr, num_cols * sizeof(float), cudaMemcpyHostToDevice));

        cusparseDnVecDescr_t vecX, vecY;
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, num_cols, d_buf_x, CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, num_rows, d_buf_b, CUDA_R_32F));

        float alpha = 1.0f;
        float beta = 0.0f;

        // 获取缓冲区大小
        if (dBuffer_spmv == nullptr) {
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize_spmv));
            CUDA_CHECK(cudaMalloc(&dBuffer_spmv, bufferSize_spmv));
        }

        CUSPARSE_CHECK(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer_spmv));

        CUDA_CHECK(cudaMemcpy(b_ptr, d_buf_b, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
    }

    void mv(torch::Tensor x, torch::Tensor b) {
        float* x_ptr = (float*)x.data_ptr<float>();
        float* b_ptr = (float*)b.data_ptr<float>();
        mv_internal(x_ptr, b_ptr);
    }

    void iter_solve(torch::Tensor b, torch::Tensor x) {
        float* h_b = (float*)b.data_ptr<float>();
        float* h_x = (float*)x.data_ptr<float>();

        // 初始化数据
        CUDA_CHECK(cudaMemcpy(d_buf_b, h_b, num_rows * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_buf_x, 0, num_cols * sizeof(float)));

        // 准备描述符
        cusparseDnVecDescr_t vecX, vecAx;
        // vecX 指向 d_buf_x
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, num_cols, d_buf_x, CUDA_R_32F));
        // vecAx 指向 d_scratch1 (用于存储 A*x 的结果)
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecAx, num_rows, d_scratch1, CUDA_R_32F));

        float spmv_alpha = 1.0f;
        float spmv_beta = 0.0f;
        float weight = 0.001f;
        
        // 确保 SpMV 缓冲区已分配
        if (dBuffer_spmv == nullptr) {
             CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &spmv_alpha, matA, vecX, &spmv_beta, vecAx, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize_spmv));
            CUDA_CHECK(cudaMalloc(&dBuffer_spmv, bufferSize_spmv));
        }

        int threads = 256;
        int blocks = (num_rows + threads - 1) / threads;

        for (int i = 0; i < 10000; i++) {
            CUSPARSE_CHECK(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &spmv_alpha, matA, vecX, &spmv_beta, vecAx, CUDA_R_32F,
                                        CUSPARSE_SPMV_ALG_DEFAULT, dBuffer_spmv));
            
            weighted_add_kernel<<<blocks, threads>>>(
                d_buf_b, d_scratch1, d_scratch2, 
                weight, -weight, num_rows
            );

            weighted_add_kernel<<<blocks, threads>>>(
                d_buf_x, d_scratch2, d_buf_x, 
                1.0f, 1.0f, num_rows
            );
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_x, d_buf_x, num_cols * sizeof(float), cudaMemcpyDeviceToHost));

        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecAx);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<csr_tensor>(m, "csr_tensor")
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int>(),
             py::arg("keys"),
             py::arg("values"),
             py::arg("row_ptr"),
             py::arg("col_ind"),
             py::arg("out_vals"),
             py::arg("num_rows"),
             py::arg("num_cols"))
        .def("mv", &csr_tensor::mv, "spmvmul")
        .def("iter_solve", &csr_tensor::iter_solve, "iter_solve");
}