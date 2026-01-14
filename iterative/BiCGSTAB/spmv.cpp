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


const uint32_t THREADS_PER_GROUP = 256;
const uint32_t SIMD_WIDTH = 32;
const uint32_t CHECK_ITERATION_NUMBER = 50;

using namespace std;
float ONE = 1.0;
float NEG_ONE = -1.0;

class csr_tensor {
    public:

        MTL::Buffer* vals;
        MTL::Buffer* row_ptr;
        MTL::Buffer* col_ind;

        MTL::Buffer* buf_x;
        MTL::Buffer* buf_b; 

        MTL::Device* device;
        MTL::CommandQueue* queue;

        MTL::ComputePipelineState* pso_spmv;
        MTL::ComputePipelineState* pso_wadd;
        MTL::ComputePipelineState* pso_freq;
        MTL::ComputePipelineState* pso_vscan;
        MTL::ComputePipelineState* pso_gscan;
        MTL::ComputePipelineState* pso_reorder;
        MTL::ComputePipelineState* pso_compress;       
        MTL::ComputePipelineState* pso_fixer;     
        MTL::ComputePipelineState* pso_iprd;
        MTL::ComputePipelineState* pso_zero;        
        MTL::ComputePipelineState* pso_divider;
        MTL::ComputePipelineState* pso_waddb;  
        MTL::ComputePipelineState* pso_up_buff;       
        MTL::ComputePipelineState* pso_update_xr;
        MTL::ComputePipelineState* pso_update_p;     
        MTL::ComputePipelineState* pso_spvmv;    

        uint num_rows;
        uint num_cols;
        uint nnz;

        void coo_to_csr_internal(
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

            MTL::CommandBuffer* cmd = queue->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();

            for (int shift = 0; shift < 64; shift += 4) { 
                uint64_t curr_mask = 0xFULL << shift;

                if ((curr_mask & actual_mask) == 0) {
                    continue;
                }

                
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
                //enc->endEncoding();

                
                
                std::swap(src_keys, dst_keys);
                std::swap(src_vals, dst_vals);
                output_buff = !output_buff;
            }

            enc->endEncoding();
            cmd->commit();
            cmd->waitUntilCompleted();

            if (output_buff) {
                std::swap(src_keys, dst_keys);
                std::swap(src_vals, dst_vals);
                dst_vals->release();
            } else {
                dst_vals->release();
            }

            MTL::Buffer* gpu_row_ptr = device->newBuffer((num_rows + 1) * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
            MTL::Buffer* gpu_col_ind = device->newBuffer(num_items * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
            
            MTL::CommandBuffer* cmd_final = queue->commandBuffer();
            
            MTL::BlitCommandEncoder* blit_fill = cmd_final->blitCommandEncoder();
            blit_fill->fillBuffer(gpu_row_ptr, NS::Range::Make(0, (num_rows + 1) * sizeof(uint32_t)), 0xFF);
            blit_fill->endEncoding();

            MTL::ComputeCommandEncoder* enc_comp = cmd_final->computeCommandEncoder();
            enc_comp->setComputePipelineState(pso_compress);
            enc_comp->setBuffer(src_keys, 0, 0);     
            enc_comp->setBuffer(gpu_row_ptr, 0, 1);    
            enc_comp->setBuffer(gpu_col_ind, 0, 2);    
            enc_comp->setBytes(&num_items, sizeof(uint32_t), 3);
            enc_comp->setBytes(&num_rows, sizeof(uint32_t), 4);
            int aligned_grid = num_groups * threads_per_group;
            enc_comp->dispatchThreads(MTL::Size::Make(aligned_grid, 1, 1), MTL::Size::Make(threads_per_group, 1, 1));
            enc_comp->memoryBarrier(MTL::BarrierScopeBuffers);

            //fixer kenel to keep everything in gpu

            enc_comp->setComputePipelineState(pso_fixer);
            enc_comp->setBuffer(gpu_row_ptr, 0, 0);
            enc_comp->setBytes(&num_rows, sizeof(uint32_t), 1);
            aligned_grid = num_groups * threads_per_group;
            enc_comp->dispatchThreads(MTL::Size::Make(aligned_grid, 1, 1), MTL::Size::Make(threads_per_group, 1, 1));
            enc_comp->endEncoding();

            cmd_final->commit();
            cmd_final->waitUntilCompleted();            
     
            this->col_ind = gpu_col_ind;
            this->vals = src_vals;
            this->row_ptr = gpu_row_ptr;
    
            buf_keys_1->release(); 
            buf_keys_2->release();
            buf_grid_counts->release(); 
            buf_grid_offsets->release();
            buf_bucket_totals->release(); 
            buf_global_offsets->release();


        }

        csr_tensor(
            torch::Tensor keys,       
            torch::Tensor values,  
            torch::Tensor row_ptr,   
            torch::Tensor col_ind,    
            torch::Tensor out_vals,   
            int num_rows,
            int num_cols  
        ) {
            this->device = MTL::CreateSystemDefaultDevice();
            this->queue = device->newCommandQueue();
            loadPipelines();

            uint64_t* keys_ptr = (uint64_t*)keys.data_ptr<int64_t>();
            float* vals_ptr = (float*)values.data_ptr<float>();
            
            uint32_t* row_ptr_ptr = (uint32_t*)row_ptr.data_ptr<int32_t>();
            uint32_t* col_ind_ptr = (uint32_t*)col_ind.data_ptr<int32_t>();
            float* out_vals_ptr = (float*)out_vals.data_ptr<float>();

            this->num_rows = num_rows;
            this->num_cols = num_cols;
            this->nnz = keys.size(0);

            this->buf_x = this->device->newBuffer(this->num_cols * sizeof(float), MTL::ResourceStorageModePrivate);
            this->buf_b = this->device->newBuffer(this->num_rows * sizeof(float), MTL::ResourceStorageModePrivate);

            coo_to_csr_internal(
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

        void loadPipelines() {
            NS::Error* error = nullptr;
        
            NS::String* libraryPath = NS::String::string("./spmv.metallib", NS::UTF8StringEncoding);
            MTL::Library* library = device->newLibrary(libraryPath, &error);
            
            if (!library) {
                std::cerr << "Failed to load library: " << error->localizedDescription()->utf8String() << std::endl;
                return;
            }

            auto loadKernel = [&](const char* name) -> MTL::ComputePipelineState* {
                NS::String* nsName = NS::String::string(name, NS::UTF8StringEncoding);
                MTL::Function* fn = library->newFunction(nsName);
                if (!fn) std::cerr << "Couldn't find function for " << name << "\n";
                MTL::ComputePipelineState* pso = device->newComputePipelineState(fn, &error);
                if (!pso) std::cerr << "Error creating PSO for " << name << ": " << error->localizedDescription()->utf8String() << std::endl;
                fn->release();
                nsName->release();
                return pso;
            };

            this->pso_spmv = loadKernel("spmv_op");
            this->pso_wadd = loadKernel("weighted_add");
            this->pso_iprd = loadKernel("inner_product");
            this->pso_zero = loadKernel("zero_out");
            this->pso_divider = loadKernel("divides");
            //this->pso_waddb = loadKernel("weighted_add_buffer");
            this->pso_up_buff = loadKernel("iter_update_buffer");
            this->pso_update_xr = loadKernel("update_xr");
            this->pso_update_p = loadKernel("update_p");
            this->pso_spvmv = loadKernel("fused_spvmv");
            
            library->release();

            libraryPath = NS::String::string("./coo_csr.metallib", NS::UTF8StringEncoding);
            library = device->newLibrary(libraryPath, &error);

            if (!library) {
                std::cerr << "Failed to load library: " << error->localizedDescription()->utf8String() << std::endl;
                return;
            }

            this->pso_freq = loadKernel("radix_frequencies");
            this->pso_vscan = loadKernel("vertical_scan");
            this->pso_gscan = loadKernel("scan_histogram");
            this->pso_reorder = loadKernel("reorder");
            this->pso_compress = loadKernel("coo_to_csr_compress");
            this->pso_fixer = loadKernel("row_fixer");
                    
            
            library->release();
        }

        ~csr_tensor() {
            this->pso_spmv->release();
            this->pso_compress->release();
            this->pso_freq->release();
            this->pso_gscan->release();
            this->pso_reorder->release();
            this->pso_fixer->release();
            this->vals->release();
            this->col_ind->release();
            this->row_ptr->release();
            this->queue->release();
            this->device->release();
        }

        void mv_buffer(
            MTL::Buffer* x,
            MTL::Buffer* b,
            MTL::CommandBuffer* cmd, 
            MTL::ComputeCommandEncoder* enc
        ) {
            enc->setComputePipelineState(this->pso_spmv);
            enc->setBuffer(this->row_ptr, 0, 0);
            enc->setBuffer(this->col_ind, 0, 1);
            enc->setBuffer(this->vals, 0, 2);
            enc->setBuffer(x, 0, 3);
            enc->setBuffer(b, 0, 4);
            enc->setBytes(&this->num_rows, sizeof(uint32_t), 5);
            enc->setBytes(&this->num_cols, sizeof(uint32_t), 6);

            uint32_t total_threads_needed = SIMD_WIDTH * this->num_rows;


            enc->dispatchThreads(MTL::Size::Make(total_threads_needed, 1, 1), MTL::Size::Make(THREADS_PER_GROUP, 1, 1));
        }


        void spvmv(
            MTL::Buffer* x,
            MTL::Buffer* y,
            MTL::Buffer* z,
            MTL::Buffer* ret,
            MTL::CommandBuffer* cmd, 
            MTL::ComputeCommandEncoder* enc
        ) {
            enc->setComputePipelineState(this->pso_spvmv);
            enc->setBuffer(this->row_ptr, 0, 0);
            enc->setBuffer(this->col_ind, 0, 1);
            enc->setBuffer(this->vals, 0, 2);
            enc->setBuffer(x, 0, 3);
            enc->setBuffer(y, 0, 4);
            enc->setBuffer(z, 0, 5);
            enc->setBuffer(ret, 0, 6);
            enc->setBytes(&this->num_rows, sizeof(uint32_t), 7);
            enc->setBytes(&this->num_cols, sizeof(uint32_t), 8);

            uint32_t total_threads_needed = SIMD_WIDTH * this->num_rows;


            enc->dispatchThreads(MTL::Size::Make(total_threads_needed, 1, 1), MTL::Size::Make(THREADS_PER_GROUP, 1, 1));
        }

        void inner_product(
            MTL::Buffer* a,
            MTL::Buffer* b,
            MTL::Buffer* out,
            uint total_size,
            MTL::CommandBuffer* cmd,
            MTL::ComputeCommandEncoder* enc
        ) {
           
            // enc->setComputePipelineState(this->pso_zero);
            // enc->setBuffer(out, 0, 0);
            // enc->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));


            //enc->memoryBarrier(MTL::BarrierScopeBuffers);
            enc->setComputePipelineState(this->pso_iprd);
            enc->setBuffer(a, 0, 0);
            enc->setBuffer(b, 0, 1);
            enc->setBuffer(out, 0, 2);
            enc->setBytes(&total_size, sizeof(uint), 3);

            uint32_t total_threads_needed = total_size;

            enc->dispatchThreads(MTL::Size::Make(total_threads_needed, 1, 1), MTL::Size::Make(THREADS_PER_GROUP, 1, 1));
        }


        void mv_internal(
            float* x,
            float* b
        ) {

            MTL::Buffer* stage_x = this->device->newBuffer(x, num_cols * sizeof(float), MTL::ResourceStorageModeShared);
            
            MTL::CommandBuffer* cmd_blit = queue->commandBuffer();
            MTL::BlitCommandEncoder* blit_init = cmd_blit->blitCommandEncoder();

            blit_init->copyFromBuffer(stage_x, 0, this->buf_x, 0, num_cols * sizeof(float));
            blit_init->endEncoding();
            cmd_blit->commit();
            cmd_blit->waitUntilCompleted();

            stage_x->release();

            MTL::CommandBuffer* cmd = queue->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();

            enc->setComputePipelineState(this->pso_spmv);
            enc->setBuffer(this->row_ptr, 0, 0);
            enc->setBuffer(this->col_ind, 0, 1);
            enc->setBuffer(this->vals, 0, 2);
            enc->setBuffer(this->buf_x, 0, 3);
            enc->setBuffer(this->buf_b, 0, 4);
            enc->setBytes(&this->num_rows, sizeof(uint32_t), 5);
            enc->setBytes(&this->num_cols, sizeof(uint32_t), 6);

            uint32_t total_threads_needed = SIMD_WIDTH * this->num_rows;


            enc->dispatchThreads(MTL::Size::Make(total_threads_needed, 1, 1), MTL::Size::Make(THREADS_PER_GROUP, 1, 1));

            enc->endEncoding();
            cmd->commit();
            cmd->waitUntilCompleted();

            MTL::Buffer* stage_b = this->device->newBuffer(this->num_rows * sizeof(float), MTL::ResourceStorageModeShared);

            MTL::CommandBuffer* cmd_last = queue->commandBuffer();
            MTL::BlitCommandEncoder* blit_last = cmd_last->blitCommandEncoder();

            blit_last->copyFromBuffer(this->buf_b, 0, stage_b, 0, num_rows * sizeof(float));
            blit_last->endEncoding();
            cmd_last->commit();
            cmd_last->waitUntilCompleted();

            memcpy(b, stage_b->contents(), (num_rows) * sizeof(float));

            stage_b->release();

        }

        void mv(
            torch::Tensor x,
            torch::Tensor b
        ) {
            float* x_vals = (float*) x.data_ptr<float>();
            float* b_vals = (float*) b.data_ptr<float>();

            mv_internal(
                x_vals,
                b_vals
            );
        }

       

        void iter_solve(
            torch::Tensor b,
            torch::Tensor x
        ) {
            float* x_vals = (float*) x.data_ptr<float>();
            float* b_vals = (float*) b.data_ptr<float>();

            iterative_op(
                x_vals,
                b_vals
            );
        }



        void zero(
            MTL::Buffer* x,
            MTL::Buffer* y,
            MTL::CommandBuffer* cmd,
            MTL::ComputeCommandEncoder* enc
        ) {
            enc->setComputePipelineState(this->pso_zero);
            enc->setBuffer(x, 0, 0);
            enc->setBuffer(y, 0, 1);
            enc->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(THREADS_PER_GROUP, 1, 1));
        }

        void iterative_op(
            float* x_vals,
            float* b_vals
        ) {
            MTL::Buffer* x = this->device->newBuffer(this->num_cols * sizeof(float), MTL::ResourceStorageModePrivate);
            MTL::Buffer* stage_b = this->device->newBuffer(b_vals, this->num_rows*sizeof(float), MTL::ResourceStorageModeShared);
            MTL::Buffer* r = device->newBuffer(this->num_rows * sizeof(float), MTL::ResourceStorageModePrivate);
            MTL::Buffer* p = device->newBuffer(this->num_rows * sizeof(float), MTL::ResourceStorageModePrivate);
            MTL::Buffer* Ap = device->newBuffer(this->num_rows * sizeof(float), MTL::ResourceStorageModePrivate);

            MTL::Buffer* r_norm_old = this->device->newBuffer(sizeof(float), MTL::ResourceStorageModePrivate);
            MTL::Buffer* r_norm_new = this->device->newBuffer(sizeof(float), MTL::ResourceStorageModePrivate);
            MTL::Buffer* p_A_p = this->device->newBuffer(sizeof(float), MTL::ResourceStorageModePrivate);

            MTL::CommandBuffer* cmd_blit = queue->commandBuffer();
            MTL::BlitCommandEncoder* blit_init = cmd_blit->blitCommandEncoder();

            blit_init->fillBuffer(x, NS::Range::Make(0, this->num_cols * sizeof(float)), 0);
            blit_init->copyFromBuffer(stage_b, 0, r, 0, this->num_rows * sizeof(float));
            blit_init->copyFromBuffer(stage_b, 0, p, 0, this->num_rows * sizeof(float));
            blit_init->endEncoding();
            cmd_blit->commit();
            cmd_blit->waitUntilCompleted();
            
            stage_b->release();

            MTL::CommandBuffer* cmd_pre = queue->commandBuffer();
            MTL::ComputeCommandEncoder* enc_pre = cmd_pre->computeCommandEncoder();

            zero(r_norm_old, r_norm_new, cmd_pre, enc_pre);
            zero(r_norm_old, p_A_p, cmd_pre, enc_pre);

            enc_pre->memoryBarrier(MTL::BarrierScopeBuffers);

            inner_product(r, r, r_norm_old, this->num_rows, cmd_pre, enc_pre);
            enc_pre->endEncoding();
            cmd_pre->commit();
            cmd_pre->waitUntilCompleted();

            MTL::CommandBuffer* cmd = queue->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();

            MTL::Buffer* curr_r_norm = r_norm_old;
            MTL::Buffer* next_r_norm = r_norm_new;

            int counter = 0;


            int iter_number = (1000 < this->num_rows) ? 1000 : this->num_rows;
            for (int i = 0; i < iter_number; i++) {

                zero(p_A_p, next_r_norm, cmd, enc);
                enc->memoryBarrier(MTL::BarrierScopeBuffers);

                spvmv(p, p, Ap, p_A_p, cmd, enc);
                enc->memoryBarrier(MTL::BarrierScopeBuffers);

                enc->setComputePipelineState(pso_update_xr);
                enc->setBuffer(x, 0, 0);
                enc->setBuffer(r, 0, 1);
                enc->setBuffer(p, 0, 2);
                enc->setBuffer(Ap, 0, 3);
                enc->setBuffer(curr_r_norm, 0, 4); 
                enc->setBuffer(p_A_p, 0, 5);   
                enc->setBuffer(next_r_norm, 0, 6); 
                enc->setBytes(&this->num_rows, sizeof(uint), 7);
                enc->dispatchThreads(MTL::Size::Make(this->num_rows, 1, 1), MTL::Size::Make(THREADS_PER_GROUP, 1, 1));
                enc->memoryBarrier(MTL::BarrierScopeBuffers);

                enc->setComputePipelineState(this->pso_update_p);
                enc->setBuffer(p, 0, 0);
                enc->setBuffer(r, 0, 1);
                enc->setBuffer(curr_r_norm, 0, 2); 
                enc->setBuffer(next_r_norm, 0, 3); 
                enc->setBytes(&this->num_rows, sizeof(uint), 4);
                enc->dispatchThreads(MTL::Size::Make(this->num_rows, 1, 1), MTL::Size::Make(THREADS_PER_GROUP, 1, 1));
                enc->memoryBarrier(MTL::BarrierScopeBuffers);

                if (i % CHECK_ITERATION_NUMBER == 0) {
                    //blit back curr_r_norm
                    enc->endEncoding();
                    cmd->commit();
                    cmd->waitUntilCompleted();

                    MTL::Buffer* stage_r_norm = this->device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
                    
                    cmd = queue->commandBuffer();
                    MTL::BlitCommandEncoder* blit_check = cmd->blitCommandEncoder();
                    blit_check->copyFromBuffer(curr_r_norm, 0, stage_r_norm, 0, sizeof(float));
                    blit_check->endEncoding();
                    cmd->commit();
                    cmd->waitUntilCompleted();

                    if (*(float*) stage_r_norm->contents() < 1e-5) {
                        cmd = queue->commandBuffer();
                        enc = cmd->computeCommandEncoder();

                        break;
                        
                    }


                    cmd = queue->commandBuffer();
                    enc = cmd->computeCommandEncoder();

                }

                std::swap(curr_r_norm, next_r_norm);
                counter++;
            }

            enc->endEncoding();
            cmd->commit();
            cmd->waitUntilCompleted();

            r->release();
            p->release();
            Ap->release();
            r_norm_old->release();
            r_norm_new->release();
            p_A_p->release();

            //cout << "i hate ts \n";


            MTL::Buffer* stage_x = this->device->newBuffer(this->num_cols * sizeof(float), MTL::ResourceStorageModeShared);

            MTL::CommandBuffer* cmd_last = queue->commandBuffer();
            MTL::BlitCommandEncoder* blit_last = cmd_last->blitCommandEncoder();

            blit_last->copyFromBuffer(x, 0, stage_x, 0, this->num_cols * sizeof(float));
            blit_last->endEncoding();
            cmd_last->commit();
            cmd_last->waitUntilCompleted();

            memcpy(x_vals, stage_x->contents(), (this->num_cols) * sizeof(float));


            stage_x->release();

        }

};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<csr_tensor>(m, "csr_tensor")
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int>(),
             "CSR Tensor Constructor",
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