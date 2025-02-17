#ifndef NNET_HEPT_SIMPLIFIED_H_
#define NNET_HEPT_SIMPLIFIED_H_

#include "hls_math.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace nnet {

struct hept_simplified_config {
    // Lookup table sizes
    static const unsigned exp_table_size = 1024;
    static const int exp_table_min = -8;
    static const int exp_table_max = 0;

    // Internal data type definitions
    typedef ap_fixed<16, 6> accum_t;
    typedef ap_fixed<16, 0> exp_table_t;
    typedef void dense_conf_qk;
    typedef void transpose_conf_qk;

    // Layer Sizes
    static const unsigned n_heads = 5;
    static const unsigned batch_size = 5;
    static const unsigned seq_len = 5;
    static const unsigned dim_per_head = 5;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned strategy = latency;
    static const unsigned reuse_factor = 1;
    static const unsigned parallelization_factor = 16;
    static const bool store_weights_in_bram = false;

    // Product function to use
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template <typename CONFIG_T, int N_TABLE> void init_exp_table(typename CONFIG_T::exp_table_t table_out[N_TABLE]) {
    float step = (float)(CONFIG_T::exp_table_max - CONFIG_T::exp_table_min) / (float)(N_TABLE);
    for (int i = 0; i < N_TABLE; i++) {
        table_out[i] = (typename CONFIG_T::exp_table_t)(std::exp(CONFIG_T::exp_table_min + step * i));
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pairwise_dist_sq_rbf(data_T query[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::dim_per_head],
                      data_T key[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::dim_per_head],
                      res_T output[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::seq_len]) {
    // Initialize the exponentiation lookup table
    #ifdef __HLS_SYN__
        bool exp_table_initialized = false;
        typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    #else
        static bool exp_table_initialized = false;
        static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    #endif
    if (!exp_table_initialized) {
        nnet::init_exp_table<CONFIG_T, CONFIG_T::exp_table_size>(exp_table);
        exp_table_initialized = true;
    }
    static const unsigned exp_table_range_inv = CONFIG_T::exp_table_size / (CONFIG_T::exp_table_max - CONFIG_T::exp_table_min);

    #pragma HLS ARRAY_PARTITION variable=query complete
    #pragma HLS ARRAY_PARTITION variable=key complete
    #pragma HLS ARRAY_PARTITION variable=output complete

    constexpr unsigned B = CONFIG_T::batch_size * CONFIG_T::n_heads;
    constexpr unsigned N = CONFIG_T::seq_len;
    constexpr unsigned D = CONFIG_T::dim_per_head;

    const typename CONFIG_T::accum_t negative_half = -0.5;
    for (unsigned b = 0; b < B; b++) {
        #pragma HLS UNROLL factor=CONFIG_T::parallelization_factor
        for (unsigned nq = 0; nq < N; nq++) {
            #pragma HLS UNROLL
            for (unsigned nk = 0; nk < N; nk++) {
                #pragma HLS UNROLL
                typename CONFIG_T::accum_t sum = 0;
                for (unsigned d = 0; d < D; d++) {
                    #pragma HLS UNROLL
                    sum += 
                        (query[b * N * D + nq * D + d] - key[b * N * D + nk * D + d]) * 
                        (query[b * N * D + nq * D + d] - key[b * N * D + nk * D + d]);
                }
                sum *= negative_half;
                int index = (sum + CONFIG_T::exp_table_max - CONFIG_T::exp_table_min) * exp_table_range_inv;
                if (index < 0) index = 0;
                if (index > CONFIG_T::exp_table_size - 1) index = CONFIG_T::exp_table_size - 1;
                output[b * N * N + nq * N + nk] = exp_table[index];
            }
        }
    }
}


template <class data_T, class res_T, typename CONFIG_T>
void hept_simplified(data_T query[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::dim_per_head],
          data_T key[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::dim_per_head],
          res_T output[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::seq_len]) {
    nnet::pairwise_dist_sq_rbf<data_T, res_T, CONFIG_T>(query, key, output);
}

} // namespace nnet

#endif
