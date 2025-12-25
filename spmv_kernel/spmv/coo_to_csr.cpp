#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <Metal/Metal.hpp>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <algorithm> 

class coo_to_csr_converter {
public:
    MTL::Device* device;
    MTL::CommandQueue* queue;
    
    MTL::ComputePipelineState* pso_freq;
    MTL::ComputePipelineState* pso_vscan;
    MTL::ComputePipelineState* pso_gscan;
    MTL::ComputePipelineState* pso_reorder;
    MTL::ComputePipelineState* pso_compress;

    coo_to_csr_converter() {
        device = MTL::CreateSystemDefaultDevice();
        queue = device->newCommandQueue();
        loadPipelines();
    }

    ~coo_to_csr_converter() {
        pso_freq->release();
        pso_vscan->release();
        pso_gscan->release();
        pso_reorder->release();
        pso_compress->release();
        queue->release();
        device->release();
    }

    void loadPipelines() {
        NS::Error* error = nullptr;
        
        NS::String* libraryPath = NS::String::string("./shaders.metallib", NS::UTF8StringEncoding);
        MTL::Library* library = device->newLibrary(libraryPath, &error);
        
        if (!library) {
            std::cerr << "Failed to load library: " << error->localizedDescription()->utf8String() << std::endl;
            return;
        }

        auto loadKernel = [&](const char* name) -> MTL::ComputePipelineState* {
            NS::String* nsName = NS::String::string(name, NS::UTF8StringEncoding);
            MTL::Function* fn = library->newFunction(nsName);
            MTL::ComputePipelineState* pso = device->newComputePipelineState(fn, &error);
            if (!pso) std::cerr << "Error creating PSO for " << name << ": " << error->localizedDescription()->utf8String() << std::endl;
            fn->release();
            nsName->release();
            return pso;
        };

        pso_freq = loadKernel("radix_frequencies");
        pso_vscan = loadKernel("vertical_scan");
        pso_gscan = loadKernel("scan_histogram");
        pso_reorder = loadKernel("reorder");
        pso_compress = loadKernel("coo_to_csr_compress");
        
        library->release();
    }

   void coo_to_csr(
        uint64_t* packed_keys_ptr,    
        float* values_ptr,         
        uint32_t num_items,          
        uint32_t num_rows, 
        uint32_t num_cols,           
        uint32_t* out_row_ptr,        
        uint32_t* out_col_ind,        
        float* out_values      
    ) {

        uint64_t max_cols_bits = 32 - __builtin_clz(num_cols);
        uint64_t max_rows_bits = 32 - __builtin_clz(num_rows);

        uint64_t actual_mask = (1ULL << max_cols_bits) - 1;
        actual_mask |= (((1ULL << max_rows_bits) - 1) << 32);

        MTL::Buffer* buf_keys_1 = device->newBuffer(num_items * sizeof(uint64_t), MTL::ResourceStorageModePrivate);
        MTL::Buffer* buf_keys_2 = device->newBuffer(num_items * sizeof(uint64_t), MTL::ResourceStorageModePrivate);
        MTL::Buffer* buf_vals_1 = device->newBuffer(num_items * sizeof(float), MTL::ResourceStorageModePrivate);
        MTL::Buffer* buf_vals_2 = device->newBuffer(num_items * sizeof(float), MTL::ResourceStorageModePrivate);

        MTL::Buffer* stage_keys_in = device->newBuffer(packed_keys_ptr, num_items * sizeof(uint64_t), MTL::ResourceStorageModeShared);
        MTL::Buffer* stage_vals_in = device->newBuffer(values_ptr, num_items * sizeof(float), MTL::ResourceStorageModeShared);
        
        MTL::CommandBuffer* cmd_init = queue->commandBuffer();
        MTL::BlitCommandEncoder* blit_init = cmd_init->blitCommandEncoder();
        blit_init->copyFromBuffer(stage_keys_in, 0, buf_keys_1, 0, num_items * sizeof(uint64_t));
        blit_init->copyFromBuffer(stage_vals_in, 0, buf_vals_1, 0, num_items * sizeof(float));
        blit_init->endEncoding();
        cmd_init->commit();
        cmd_init->waitUntilCompleted();
        
        stage_keys_in->release(); 
        stage_vals_in->release(); 

        int threads_per_group = 32; 
        int num_groups = (num_items + threads_per_group - 1) / threads_per_group;
        int num_buckets = 16; 

        MTL::Buffer* buf_grid_counts = device->newBuffer(num_groups * num_buckets * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
        MTL::Buffer* buf_grid_offsets = device->newBuffer(num_groups * num_buckets * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
        MTL::Buffer* buf_bucket_totals = device->newBuffer(num_buckets * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
        MTL::Buffer* buf_global_offsets = device->newBuffer(num_buckets * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
        
        MTL::Buffer* src_keys = buf_keys_1;
        MTL::Buffer* dst_keys = buf_keys_2;
        MTL::Buffer* src_vals = buf_vals_1;
        MTL::Buffer* dst_vals = buf_vals_2;

        bool output_buff = 0;

        for (int shift = 0; shift < 64; shift += 4) { 
            uint64_t curr_mask = 0xFULL << shift;

            if ((curr_mask & actual_mask) == 0) {
                continue;
            }

            MTL::CommandBuffer* cmd = queue->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
            int aligned_grid_w = num_groups * threads_per_group;
            
            //freq
            enc->setComputePipelineState(pso_freq);
            enc->setBuffer(src_keys, 0, 0);
            enc->setBuffer(buf_grid_counts, 0, 1);
            enc->setBytes(&num_items, sizeof(uint32_t), 2);
            enc->setBytes(&shift, sizeof(int), 3);
            enc->dispatchThreads(MTL::Size::Make(aligned_grid_w, 1, 1), MTL::Size::Make(threads_per_group, 1, 1));
            enc->memoryBarrier(MTL::BarrierScopeBuffers);

            //vscan
            enc->setComputePipelineState(pso_vscan);
            enc->setBuffer(buf_grid_counts, 0, 0);
            enc->setBuffer(buf_grid_offsets, 0, 1);
            enc->setBuffer(buf_bucket_totals, 0, 2);
            enc->setBytes(&num_groups, sizeof(uint32_t), 3);
            enc->dispatchThreads(MTL::Size::Make(num_buckets, 1, 1), MTL::Size::Make(num_buckets, 1, 1));
            enc->memoryBarrier(MTL::BarrierScopeBuffers);
            
            //scan
            enc->setComputePipelineState(pso_gscan);
            enc->setBuffer(buf_bucket_totals, 0, 0);
            enc->setBuffer(buf_global_offsets, 0, 1);
            enc->dispatchThreads(MTL::Size::Make(num_buckets, 1, 1), MTL::Size::Make(num_buckets, 1, 1));
            enc->memoryBarrier(MTL::BarrierScopeBuffers);

            //reorder
            enc->setComputePipelineState(pso_reorder);
            enc->setBuffer(src_keys, 0, 0);
            enc->setBuffer(dst_keys, 0, 1);
            enc->setBuffer(src_vals, 0, 2); 
            enc->setBuffer(dst_vals, 0, 3); 
            enc->setBuffer(buf_grid_offsets, 0, 4);
            enc->setBuffer(buf_global_offsets, 0, 5);
            enc->setBytes(&num_items, sizeof(uint32_t), 6);
            enc->setBytes(&shift, sizeof(int), 7);
            enc->dispatchThreads(MTL::Size::Make(aligned_grid_w, 1, 1), MTL::Size::Make(threads_per_group, 1, 1));
            enc->endEncoding();

            cmd->commit();
            cmd->waitUntilCompleted();
            
            std::swap(src_keys, dst_keys);
            std::swap(src_vals, dst_vals);
            output_buff = !output_buff;
        }

        if (output_buff) {
            std::swap(src_keys, dst_keys);
            std::swap(src_vals, dst_vals);
        }

        MTL::Buffer* gpu_row_ptr = device->newBuffer((num_rows + 1) * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
        MTL::Buffer* gpu_col_ind = device->newBuffer(num_items * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
        
        MTL::CommandBuffer* cmd_final = queue->commandBuffer();
        
        MTL::BlitCommandEncoder* blit_fill = cmd_final->blitCommandEncoder();
        uint32_t fill_val = 0xFFFFFFFF;

        blit_fill->fillBuffer(gpu_row_ptr, NS::Range::Make(0, (num_rows + 1) * sizeof(uint32_t)), 0xFF);
        blit_fill->endEncoding();

        MTL::ComputeCommandEncoder* enc = cmd_final->computeCommandEncoder();
        enc->setComputePipelineState(pso_compress);
        enc->setBuffer(src_keys, 0, 0);     
        enc->setBuffer(gpu_row_ptr, 0, 1);    
        enc->setBuffer(gpu_col_ind, 0, 2);    
        enc->setBytes(&num_items, sizeof(uint32_t), 3);
        enc->setBytes(&num_rows, sizeof(uint32_t), 4);
        int aligned_grid = num_groups * threads_per_group;
        enc->dispatchThreads(MTL::Size::Make(aligned_grid, 1, 1), MTL::Size::Make(threads_per_group, 1, 1));
        enc->endEncoding();

        MTL::Buffer* stage_row_ptr = device->newBuffer((num_rows + 1) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        MTL::Buffer* stage_col_ind = device->newBuffer(num_items * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        MTL::Buffer* stage_vals_out = device->newBuffer(num_items * sizeof(float), MTL::ResourceStorageModeShared);

        MTL::BlitCommandEncoder* blit_down = cmd_final->blitCommandEncoder();
        blit_down->copyFromBuffer(gpu_row_ptr, 0, stage_row_ptr, 0, (num_rows + 1) * sizeof(uint32_t));
        blit_down->copyFromBuffer(gpu_col_ind, 0, stage_col_ind, 0, num_items * sizeof(uint32_t));
        blit_down->copyFromBuffer(src_vals, 0, stage_vals_out, 0, num_items * sizeof(float));
        blit_down->endEncoding();

        cmd_final->commit();
        cmd_final->waitUntilCompleted();
        
        memcpy(out_row_ptr, stage_row_ptr->contents(), (num_rows + 1) * sizeof(uint32_t));
        memcpy(out_col_ind, stage_col_ind->contents(), num_items * sizeof(uint32_t));
        memcpy(out_values,  stage_vals_out->contents(), num_items * sizeof(float));
        

        out_row_ptr[num_rows] = num_items;

        for (int i = num_rows - 1; i >= 0; --i) {
            if (out_row_ptr[i] == 0xFFFFFFFF) {
                out_row_ptr[i] = out_row_ptr[i+1];
            }
        }
        
        buf_keys_1->release(); 
        buf_keys_2->release();
        buf_vals_1->release(); 
        buf_vals_2->release();
        buf_grid_counts->release(); 
        buf_grid_offsets->release();
        buf_bucket_totals->release(); 
        buf_global_offsets->release();
        gpu_row_ptr->release(); 
        gpu_col_ind->release();
        stage_row_ptr->release(); 
        stage_col_ind->release(); 
        stage_vals_out->release();
    }
};


void coo_to_csr_wrapper(
    torch::Tensor keys,       
    torch::Tensor values,  
    torch::Tensor row_ptr,   
    torch::Tensor col_ind,    
    torch::Tensor out_vals,   
    int num_rows,
    int num_cols            
) {

    static coo_to_csr_converter converter;

    uint64_t* keys_ptr = (uint64_t*)keys.data_ptr<int64_t>();
    float* vals_ptr = (float*)values.data_ptr<float>();
    
    uint32_t* row_ptr_ptr = (uint32_t*)row_ptr.data_ptr<int32_t>();
    uint32_t* col_ind_ptr = (uint32_t*)col_ind.data_ptr<int32_t>();
    float* out_vals_ptr = (float*)out_vals.data_ptr<float>();

    converter.coo_to_csr(
        keys_ptr,
        vals_ptr,
        keys.size(0),  
        num_rows,
        num_cols,
        row_ptr_ptr,
        col_ind_ptr,
        out_vals_ptr
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("coo_csr", &coo_to_csr_wrapper, "coo to csr");
}
