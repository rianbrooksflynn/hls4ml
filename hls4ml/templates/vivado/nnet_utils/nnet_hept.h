#ifndef NNET_HEPT_H_
#define NNET_HEPT_H_

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

struct hept_config {
    // Lookup table sizes
    static const unsigned table_size = 1024;
    static const int table_min = -8;
    static const int table_max = 0;

    // Internal data type definitions
    typedef ap_fixed<16, 6> accum_t;
    typedef ap_fixed<16, 1> table_t;
    typedef void dense_conf_qk;
    typedef void dense_conf_qkv;
    typedef void transpose_conf_qk;

    // Layer Sizes
    static const unsigned n_heads = 5;
    static const unsigned n_blocks = 5;
    static const unsigned block_size = 5;
    static const unsigned dim_per_head = 5;
    static const unsigned coords_dim = 3;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned strategy = latency;
    static const unsigned reuse_factor = 1;
    static const unsigned parallelization_factor = 16;
    static const bool store_weights_in_bram = false;

    // Product function to use
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void negative_half_sum_square(
    data_T input[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)],
    res_T output[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size]) {
    // Negative one half times the sum over the last dimension of the elements squared
    #pragma HLS ARRAY_PARTITION variable=output complete
    #pragma HLS ARRAY_PARTITION variable=input complete

    const typename CONFIG_T::accum_t negative_half = -0.5;
    for (unsigned i = 0; i < CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size; i++) {
        #pragma HLS UNROLL factor=CONFIG_T::parallelization_factor
        typename CONFIG_T::accum_t sum = 0;
        for (unsigned j = 0; j < CONFIG_T::dim_per_head + CONFIG_T::coords_dim; j++) {
            #pragma HLS UNROLL
            sum += input[i * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim) + j] *
                   input[i * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim) + j];
        }
        output[i] = sum * negative_half;
    }
}

template <typename CONFIG_T, int N_TABLE> void init_exp_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    float step = (float)(CONFIG_T::table_max - CONFIG_T::table_min) / (float)(N_TABLE);
    for (int i = 0; i < N_TABLE; i++) {
        table_out[i] = (typename CONFIG_T::table_t)(std::exp(CONFIG_T::table_min + step * i));
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void add_clamp_exp(
    data_T cluster_sum[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * CONFIG_T::block_size],
    data_T q_sq_05[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size],
    data_T k_sq_05[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size],
    res_T output[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * CONFIG_T::block_size]) {
    // Add the inputs, clamp to max value 0, and exponentiate
    // Initialize the exponentiation lookup table
    #ifdef __HLS_SYN__
        bool initialized = false;
        typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    #else
        static bool initialized = false;
        static typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    #endif
    if (!initialized) {
        init_exp_table<CONFIG_T, CONFIG_T::table_size>(exp_table);
        initialized = true;
    }
    static const unsigned inv_table_range = CONFIG_T::table_size / (CONFIG_T::table_max - CONFIG_T::table_min);
    
    #pragma HLS ARRAY_PARTITION variable=output complete
    #pragma HLS ARRAY_PARTITION variable=cluster_sum complete
    #pragma HLS ARRAY_PARTITION variable=q_sq_05 complete
    #pragma HLS ARRAY_PARTITION variable=k_sq_05 complete

    constexpr unsigned A = CONFIG_T::n_heads * CONFIG_T::n_blocks;
    constexpr unsigned B = CONFIG_T::block_size;

    for (unsigned a = 0; a < A; a++) {
        #pragma HLS UNROLL factor=CONFIG_T::parallelization_factor
        for (unsigned b = 0; b < B; b++) {
            #pragma HLS UNROLL
            for (unsigned b1 = 0; b1 < B; b1++) {
                #pragma HLS UNROLL
                typename CONFIG_T::accum_t sum = (typename CONFIG_T::accum_t)cluster_sum[a * B * B + b * B + b1] + 
                    (typename CONFIG_T::accum_t)q_sq_05[a * B + b] + 
                    (typename CONFIG_T::accum_t)k_sq_05[a * B + b1];
                typename CONFIG_T::accum_t clamp_out = sum < 0 ? sum : (typename CONFIG_T::accum_t)0.0;
                int index = (clamp_out + CONFIG_T::table_max - CONFIG_T::table_min) * inv_table_range;
                if (index < 0) index = 0;
                if (index > CONFIG_T::table_size - 1) index = CONFIG_T::table_size - 1;
                output[a * B * B + b * B + b1] = exp_table[index];
            }
        }
    }
}


template <class data_T, class res_T, typename CONFIG_T>
void qk_einsum(
    data_T query[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)],
    data_T key[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)],
    res_T output[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * CONFIG_T::block_size]) {
    data_T key_transpose[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)];
    res_T out_buffer[CONFIG_T::block_size];
    typename CONFIG_T::dense_conf_qk::bias_t biases[CONFIG_T::block_size];
    nnet::fill_zero<typename CONFIG_T::dense_conf_qk::bias_t, CONFIG_T::block_size>(biases);

    #pragma HLS ARRAY_PARTITION variable=query complete
    #pragma HLS ARRAY_PARTITION variable=key_transpose complete
    #pragma HLS ARRAY_PARTITION variable=output complete

    nnet::transpose<data_T, data_T, typename CONFIG_T::transpose_conf_qk>(key, key_transpose);

    constexpr unsigned A = CONFIG_T::n_heads * CONFIG_T::n_blocks;
    constexpr unsigned B = CONFIG_T::block_size;
    constexpr unsigned C = CONFIG_T::dim_per_head + CONFIG_T::coords_dim;

    for (unsigned a = 0; a < A; a++) {
        #pragma HLS UNROLL factor=CONFIG_T::parallelization_factor
        for (unsigned b = 0; b < B; b++) {
            #pragma HLS UNROLL
            dense<data_T, res_T, typename CONFIG_T::dense_conf_qk>(&query[(a * B * C + b * C)], out_buffer,
                                                                &key_transpose[(a * B * C)], biases);
            for (unsigned b1 = 0; b1 < B; b1++) {
                #pragma HLS UNROLL
                output[(a * B * B + b * B + b1)] = out_buffer[b1];
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void qk_v_einsum(
    res_T qk[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * CONFIG_T::block_size],
    data_T value[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * CONFIG_T::dim_per_head],
    res_T output[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * CONFIG_T::dim_per_head]) {
    res_T out_buffer[CONFIG_T::dim_per_head];
    typename CONFIG_T::dense_conf_qkv::bias_t biases[CONFIG_T::dim_per_head];
    nnet::fill_zero<typename CONFIG_T::dense_conf_qkv::bias_t, CONFIG_T::dim_per_head>(biases);

    #pragma HLS ARRAY_PARTITION variable=qk complete
    #pragma HLS ARRAY_PARTITION variable=value complete
    #pragma HLS ARRAY_PARTITION variable=output complete

    constexpr unsigned A = CONFIG_T::n_heads * CONFIG_T::n_blocks;
    constexpr unsigned B = CONFIG_T::block_size;
    constexpr unsigned C = CONFIG_T::dim_per_head;

    for (unsigned a = 0; a < A; a++) {
        #pragma HLS UNROLL factor=CONFIG_T::parallelization_factor
        for (unsigned b = 0; b < B; b++) {
            #pragma HLS UNROLL
            dense<res_T, res_T, typename CONFIG_T::dense_conf_qkv>(&qk[(a * B * B + b * B)], out_buffer,
                                                                &value[(a * B * C)], biases);
            for (unsigned c = 0; c < C; c++) {
                #pragma HLS UNROLL
                output[(a * B * C + b * C + c)] = out_buffer[c];
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void hept(data_T query[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)],
          data_T key[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)],
          data_T value[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * CONFIG_T::dim_per_head],
          res_T output[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * CONFIG_T::dim_per_head]) {
    res_T q_sq_05[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size];
    res_T k_sq_05[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size];
    res_T cluster_sum[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * CONFIG_T::block_size];
    res_T qk[CONFIG_T::n_heads * CONFIG_T::n_blocks * CONFIG_T::block_size * CONFIG_T::block_size];

    nnet::negative_half_sum_square<data_T, res_T, CONFIG_T>(query, q_sq_05);
    nnet::negative_half_sum_square<data_T, res_T, CONFIG_T>(key, k_sq_05);
    nnet::qk_einsum<data_T, res_T, CONFIG_T>(query, key, cluster_sum);
    nnet::add_clamp_exp<res_T, res_T, CONFIG_T>(cluster_sum, q_sq_05, k_sq_05, qk);
    nnet::qk_v_einsum<data_T, res_T, CONFIG_T>(qk, value, output);
}

} // namespace nnet

#endif
