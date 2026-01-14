//
//  coo_csr.metal
//  coo_csr
//
//  Created by Sidharth Niwas Sharma on 12/2/25.
//

#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

#define NUM_BUCKETS 16 //2^4
#define RADIX_MASK 0xF //mask for 4 bits


kernel void radix_frequencies(
    device const ulong* keys [[buffer(0)]],
    device uint* global_counts [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant int& shift [[buffer(3)]],
    uint gid [[ thread_position_in_grid ]],
    uint lid [[ thread_position_in_threadgroup ]],
    uint group_id [[ threadgroup_position_in_grid ]]
)  {
    threadgroup atomic_uint local_counts[NUM_BUCKETS];
    if (lid < NUM_BUCKETS) {
        atomic_store_explicit(&local_counts[lid], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid < n) {
        ulong key = keys[gid];
        uint bucket = (key >> shift) & RADIX_MASK;
        atomic_fetch_add_explicit(&local_counts[bucket], 1, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < NUM_BUCKETS) {
        uint local_count = atomic_load_explicit(&local_counts[lid], memory_order_relaxed);
        uint global_idx = (group_id * NUM_BUCKETS) + lid;
        
        global_counts[global_idx] = local_count; //safe (trust)
    }
}

kernel void vertical_scan(
    device const uint* grid_counts [[buffer(0)]], //input
    device uint* grid_offsets [[buffer(1)]], //output
    device uint* bucket_totals [[ buffer(2) ]], //output
    constant uint& num_groups [[ buffer(3) ]], 
    uint bucket_id [[ thread_position_in_grid ]] 
) {
    if (bucket_id >= NUM_BUCKETS) {
        return;
    }

    uint sum = 0;

    //prefix summing sums
    for (uint group = 0; group < num_groups; group++) {
        uint idx = group * NUM_BUCKETS + bucket_id;
        uint count = grid_counts[idx];
        grid_offsets[idx] = sum;
        sum += count;
    }
    bucket_totals[bucket_id] = sum;
}


kernel void scan_histogram(
    device uint* bucket_totals [[buffer(0)]], //input
    device uint* global_offsets [[buffer(1)]], //output
    uint lid [[ thread_position_in_threadgroup ]] //metadata
) {
    threadgroup uint temp[NUM_BUCKETS];
    temp[lid] = bucket_totals[lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    //hillis-steele
    for (uint offset = 1; offset < NUM_BUCKETS; offset <<= 1) {
        uint t = 0;
        if (lid >= offset) {
            t = temp[lid - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid >= offset) {
            temp[lid] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        global_offsets[0] = 0;
    } else {
        global_offsets[lid] = temp[lid - 1];
    }
}

kernel void reorder(
    device const ulong* input_keys [[buffer(0)]],
    device ulong* output_keys [[buffer(1)]],
    device const float* input_values [[buffer(2)]],
    device float* output_values [[buffer(3)]],
    device const uint* grid_offsets [[buffer(4)]], 
    device const uint* global_offsets [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant int& shift [[buffer(7)]],
    uint gid [[ thread_position_in_grid ]],
    uint group_id [[ threadgroup_position_in_grid ]],
    uint lid [[thread_position_in_threadgroup]]
) {
    if (gid >= n) return;

    ulong key = input_keys[gid];
    float value = input_values[gid];
    uint bucket = (uint)((key >> shift) & RADIX_MASK); 

    
    uint local_idx = 0;
    
    #pragma unroll
    for (uint i = 0; i < NUM_BUCKETS; ++i) {
        bool is_mine = (bucket == i);
    
        //number of threads w lid < my_lid that have same bucket
        uint rank = simd_prefix_exclusive_sum(is_mine ? 1 : 0);
        
        if (is_mine) {
            local_idx = rank;
        }
    }


    uint group_base = grid_offsets[(group_id * NUM_BUCKETS) + bucket];
    uint global_base = global_offsets[bucket];
    uint output_idx = global_base + group_base + local_idx;

    output_keys[output_idx] = key;
    output_values[output_idx] = value;
}

kernel void coo_to_csr_compress(
    device const ulong* sorted_keys [[buffer(0)]], //input
    device uint* csr_row_ptr [[buffer(1)]], //output
    device uint* csr_col_ind [[buffer(2)]], //output
    constant uint& num_nonzeros [[buffer(3)]], 
    constant uint& num_rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) 
{
    if (gid >= num_nonzeros) {
        return;
    }

    ulong key = sorted_keys[gid];
    uint row = key >> 32;
    uint col = key & 0xFFFFFFFF;

    csr_col_ind[gid] = col;

    if (gid == 0) {
        csr_row_ptr[0] = 0;
        csr_row_ptr[row] = 0;
    } else {
        ulong prevkey = sorted_keys[gid - 1];
        uint prevrow = prevkey >> 32;

        if (row != prevrow) {
            csr_row_ptr[row] = gid;
        }
    }

    if (gid == num_nonzeros - 1) {
        csr_row_ptr[num_rows] = num_nonzeros;
    }

}

kernel void row_fixer(
    device uint* csr_row_ptr [[buffer(0)]], //input and output
    constant uint& num_rows [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    //single thread for now, can fix later
    if (gid == 0) {
        for (int i = num_rows - 1; i >= 0; --i) {
            if (csr_row_ptr[i] == 0xFFFFFFFF) {
                csr_row_ptr[i] = csr_row_ptr[i+1];
            }
        }
    }
}





