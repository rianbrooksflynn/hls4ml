#ifndef NNET_HEPT_H_
#define NNET_HEPT_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"
#include <iostream>

namespace nnet {

struct hept_config {
    // Internal data type definitions
    typedef void dense_conf;
    typedef void transpose_conf;

    // Layer Sizes
    static const unsigned n_blocks = 4;
    static const unsigned block_size = 4;
    static const unsigned dim_per_head = 2;
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
void qk_einsum(
    data_T query[CONFIG_T::n_blocks * CONFIG_T::block_size * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)],
    data_T key[CONFIG_T::n_blocks * CONFIG_T::block_size * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)],
    res_T output[CONFIG_T::n_blocks * CONFIG_T::block_size * CONFIG_T::block_size]) {
    data_T key_transpose[CONFIG_T::n_blocks * CONFIG_T::block_size * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim)];
    res_T out_buffer[CONFIG_T::block_size];
    typename CONFIG_T::dense_conf::bias_t biases[CONFIG_T::block_size];
    for (auto &bias : biases) {
        bias = 0;
    }

    #pragma HLS ARRAY_PARTITION variable=query complete
    #pragma HLS ARRAY_PARTITION variable=key_transpose complete
    #pragma HLS ARRAY_PARTITION variable=output complete

    nnet::transpose<data_T, data_T, typename CONFIG_T::transpose_conf>(key, key_transpose);

    constexpr unsigned A = CONFIG_T::n_blocks;
    constexpr unsigned B = CONFIG_T::block_size;
    constexpr unsigned C = CONFIG_T::dim_per_head + CONFIG_T::coords_dim;

    for (unsigned a = 0; a < A; a++) {
        #pragma HLS UNROLL factor=CONFIG_T::parallelization_factor
        for (unsigned b = 0; b < B; b++) {
            #pragma HLS UNROLL
            dense<data_T, res_T, typename CONFIG_T::dense_conf>(&query[(a * B * C + b * C)], out_buffer,
                                                                &key_transpose[(a * B * C)], biases);
            for (unsigned c = 0; c < C; c++) {
                #pragma HLS UNROLL
                output[(a * B * B + b * B + c)] = out_buffer[c];
            }
        }
    }
}

} // namespace nnet

#endif
