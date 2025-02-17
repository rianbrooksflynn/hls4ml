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
    // Epsilon for normalization (epsilon = 2^(-eps_power))
    static const unsigned eps_power = 4;

    // Exponentiation lookup table
    static const unsigned exp_table_size = 1024;
    static const int exp_table_min = -8;
    static const int exp_table_max = 0;

    // Inversion lookup table
    static const unsigned inv_table_size = 1024;
    // Minimum value assumed to be epsilon
    static const unsigned inv_table_max = 8;

    // Internal data type definitions
    typedef ap_fixed<16, 6> accum_t;
    typedef ap_fixed<16, 0> exp_table_t;
    typedef ap_fixed<16, 10> inv_table_t;
    typedef void dense_conf;
    typedef void transpose_conf_qk;
    typedef void transpose_conf_v;
    typedef void transpose_conf_output;

    // Layer Sizes
    static const unsigned n_heads = 5;
    static const unsigned batch_size = 5;
    static const unsigned seq_len = 5;
    static const unsigned dim_per_head = 5;
    static const unsigned coords_dim = 2;

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
void transpose_qk(data_T input[CONFIG_T::n_heads * CONFIG_T::batch_size * CONFIG_T::seq_len * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)],
                   res_T output[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)]) {
    #pragma HLS ARRAY_PARTITION variable=input complete
    #pragma HLS ARRAY_PARTITION variable=output complete
    nnet::transpose<data_T, res_T, typename CONFIG_T::transpose_conf_qk>(input, output);
}

template <class data_T, class res_T, typename CONFIG_T>
void transpose_v(data_T input[CONFIG_T::n_heads * CONFIG_T::batch_size * CONFIG_T::seq_len * CONFIG_T::dim_per_head],
                   res_T output[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::dim_per_head]) {
    #pragma HLS ARRAY_PARTITION variable=input complete
    #pragma HLS ARRAY_PARTITION variable=output complete
    nnet::transpose<data_T, res_T, typename CONFIG_T::transpose_conf_v>(input, output);
}

template <class data_T, class res_T, typename CONFIG_T>
void transpose_output(data_T input[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::dim_per_head],
                      res_T output[CONFIG_T::n_heads * CONFIG_T::batch_size * CONFIG_T::seq_len * CONFIG_T::dim_per_head]) {
    #pragma HLS ARRAY_PARTITION variable=input complete
    #pragma HLS ARRAY_PARTITION variable=output complete
    nnet::transpose<data_T, res_T, typename CONFIG_T::transpose_conf_output>(input, output);
}

template <typename CONFIG_T, int N_TABLE> void init_exp_table(typename CONFIG_T::exp_table_t table_out[N_TABLE]) {
    float step = (float)(CONFIG_T::exp_table_max - CONFIG_T::exp_table_min) / (float)(N_TABLE);
    for (int i = 0; i < N_TABLE; i++) {
        table_out[i] = (typename CONFIG_T::exp_table_t)(std::exp(CONFIG_T::exp_table_min + step * i));
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pairwise_dist_sq_rbf(data_T query[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)],
                      data_T key[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)],
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
    constexpr unsigned D = CONFIG_T::dim_per_head + CONFIG_T::coords_dim;

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

template <typename CONFIG_T, int N_TABLE>
void init_invert_table(typename CONFIG_T::inv_table_t table_out[N_TABLE]) {
    float epsilon = 1.0 / (1 << CONFIG_T::eps_power);
    float max_val = CONFIG_T::inv_table_max;
    float step = max_val / (float)(N_TABLE);
    for (int i = 0; i < N_TABLE; i++) {
        table_out[i] = (typename CONFIG_T::inv_table_t)(1.0 / (epsilon + step * i));
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void mask_and_normalize(
    res_T kernel[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::seq_len],
    data_T padding_mask[CONFIG_T::batch_size * CONFIG_T::seq_len],
    res_T output[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::seq_len]) {
    // Initialize the inversion lookup table
    #ifdef __HLS_SYN__
        bool inv_table_initialized = false;
        typename CONFIG_T::inv_table_t inv_table[CONFIG_T::inv_table_size];
    #else
        static bool inv_table_initialized = false;
        static typename CONFIG_T::inv_table_t inv_table[CONFIG_T::inv_table_size];
    #endif
    if (!inv_table_initialized) {
        nnet::init_invert_table<CONFIG_T, CONFIG_T::inv_table_size>(inv_table);
        inv_table_initialized = true;
    }
    static const unsigned inv_table_range_inv = CONFIG_T::inv_table_size / CONFIG_T::inv_table_max;

    typename CONFIG_T::accum_t masked_kernel[CONFIG_T::seq_len];
    typename CONFIG_T::inv_table_t denom = 0;

    #pragma HLS ARRAY_PARTITION variable=kernel complete
    #pragma HLS ARRAY_PARTITION variable=padding_mask complete
    #pragma HLS ARRAY_PARTITION variable=output complete

    constexpr unsigned B = CONFIG_T::batch_size;
    constexpr unsigned H = CONFIG_T::n_heads;
    constexpr unsigned N = CONFIG_T::seq_len;

    for (unsigned b = 0; b < B; b++) {
        #pragma HLS UNROLL factor=CONFIG_T::parallelization_factor
        for (unsigned h = 0; h < H; h++) {
            #pragma HLS UNROLL
            for (unsigned n1 = 0; n1 < N; n1++) {
                #pragma HLS UNROLL
                typename CONFIG_T::accum_t sum = 0;
                for (unsigned n2 = 0; n2 < N; n2++) {
                    #pragma HLS UNROLL
                    masked_kernel[n2] = 
                        kernel[b * H * N * N + h * N * N + n1 * N + n2] * 
                        padding_mask[b * N + n1] * padding_mask[b * N + n2];
                    sum += masked_kernel[n2];
                }
                int index = sum * inv_table_range_inv;
                if (index < 0) index = 0;
                if (index > CONFIG_T::inv_table_size - 1) index = CONFIG_T::inv_table_size - 1;
                denom = inv_table[index];
                for (unsigned n2 = 0; n2 < N; n2++) {
                    #pragma HLS UNROLL
                    output[b * H * N * N + h * N * N + n1 * N + n2] = masked_kernel[n2] * denom;
                }
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void bmm(
    res_T qk[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::seq_len],
    data_T value[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::dim_per_head],
    res_T output[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::dim_per_head]) {
    res_T out_buffer[CONFIG_T::dim_per_head];
    typename CONFIG_T::dense_conf::bias_t biases[CONFIG_T::dim_per_head];
    nnet::fill_zero<typename CONFIG_T::dense_conf::bias_t, CONFIG_T::dim_per_head>(biases);

    #pragma HLS ARRAY_PARTITION variable=qk complete
    #pragma HLS ARRAY_PARTITION variable=value complete
    #pragma HLS ARRAY_PARTITION variable=output complete

    constexpr unsigned B = CONFIG_T::batch_size * CONFIG_T::n_heads;
    constexpr unsigned N = CONFIG_T::seq_len;
    constexpr unsigned D = CONFIG_T::dim_per_head;

    for (unsigned b = 0; b < B; b++) {
        #pragma HLS UNROLL factor=CONFIG_T::parallelization_factor
        for (unsigned n = 0; n < N; n++) {
            #pragma HLS UNROLL
            nnet::dense<res_T, res_T, typename CONFIG_T::dense_conf>(&qk[(b * N * N + n * N)], out_buffer,
                                                                &value[(b * N * D)], biases);
            for (unsigned d = 0; d < D; d++) {
                #pragma HLS UNROLL
                output[(b * N * D + n * D + d)] = out_buffer[d];
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void hept(data_T query[CONFIG_T::n_heads * CONFIG_T::batch_size * CONFIG_T::seq_len * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)],
          data_T key[CONFIG_T::n_heads * CONFIG_T::batch_size * CONFIG_T::seq_len * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)],
          data_T value[CONFIG_T::n_heads * CONFIG_T::batch_size * CONFIG_T::seq_len * CONFIG_T::dim_per_head],
          data_T padding_mask[CONFIG_T::batch_size * CONFIG_T::seq_len],
          res_T output[CONFIG_T::n_heads * CONFIG_T::batch_size * CONFIG_T::seq_len * CONFIG_T::dim_per_head]) {
    data_T q_perm[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)];
    data_T k_perm[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)];
    res_T v_perm[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::dim_per_head];
    res_T qk[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::seq_len];
    res_T qk_norm[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::seq_len];
    res_T output_perm[CONFIG_T::batch_size * CONFIG_T::n_heads * CONFIG_T::seq_len * CONFIG_T::dim_per_head];

    nnet::transpose_qk<data_T, data_T, CONFIG_T>(query, q_perm);
    nnet::transpose_qk<data_T, data_T, CONFIG_T>(key, k_perm);
    nnet::transpose_v<data_T, data_T, CONFIG_T>(value, v_perm);
    nnet::pairwise_dist_sq_rbf<data_T, res_T, CONFIG_T>(q_perm, k_perm, qk);
    nnet::mask_and_normalize<data_T, res_T, CONFIG_T>(qk, padding_mask, qk_norm);
    nnet::bmm<data_T, res_T, CONFIG_T>(qk_norm, v_perm, output_perm);
    nnet::transpose_output<res_T, res_T, CONFIG_T>(output_perm, output);
}

} // namespace nnet

#endif
