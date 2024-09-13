#ifndef NNET_NORMALDIST_H
#define NNET_NORMALDIST_H

#include "nnet_common.h"
#include "nnet_math.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "ap_int.h"

namespace nnet {

const int LFSR_BITS = 16;
const int MAX_LFSR_VAL = (1 << LFSR_BITS) - 1;
const int LOG_LUT_SIZE = 2048;
const int SQRT_LUT_SIZE = 16384;
const float SQRT_LUT_MAX_VAL = -2.0f * log(1.0f / MAX_LFSR_VAL) + 1.0E-5;

struct normaldist_config {
    // Resource reuse
    static const unsigned io_type = io_parallel;
    static const unsigned strategy = latency; 
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
};


// Initialize the lookup table for log
void init_log_lut(float log_lut[LOG_LUT_SIZE]) {
    for (int i = 1; i < LOG_LUT_SIZE; i++) {
        float x = static_cast<float>(i) / LOG_LUT_SIZE;  // x ranges from 0 to 1
        log_lut[i] = log(x);
    }
}

// Initialize the lookup table for sqrt
void init_sqrt_lut(float sqrt_lut[SQRT_LUT_SIZE]) {
    sqrt_lut[0] = 0;
    for (int i = 1; i < SQRT_LUT_SIZE; i++) {
        float x = static_cast<float>(i) * SQRT_LUT_MAX_VAL / SQRT_LUT_SIZE;  // x ranges from 0 to SQRT_LUT_MAX_VAL
        sqrt_lut[i] = sqrt(x);
    }
}

// Lookup approximate log value from the table
float log_lut_lookup(float x, float log_lut[LOG_LUT_SIZE]) {
    int index = static_cast<int>(x * LOG_LUT_SIZE);
    if (index >= LOG_LUT_SIZE) index = LOG_LUT_SIZE - 1;
    return log_lut[index];
}

// Lookup approximate sqrt value from the table
float sqrt_lut_lookup(float x, float sqrt_lut[SQRT_LUT_SIZE]) {
    int index = static_cast<int>(x * SQRT_LUT_SIZE / SQRT_LUT_MAX_VAL);
    if (index >= SQRT_LUT_SIZE) index = SQRT_LUT_SIZE - 1;
    return sqrt_lut[index];
}

// LFSR Implementation (16-bit LFSR with taps at 16, 14, 13, 11)
ap_uint<LFSR_BITS> lfsr_16bit(ap_uint<LFSR_BITS>& lfsr) {
    bool bit = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1u;
    lfsr = (lfsr >> 1) | (bit << 15);
    return lfsr;
}

// Generate a uniform random floating-point number between 0 and 1
float uniform_random(ap_uint<LFSR_BITS>& lfsr) {
    ap_uint<LFSR_BITS> rand_val = lfsr_16bit(lfsr);
    return static_cast<float>(rand_val) / MAX_LFSR_VAL;
}

// Box-Muller transform using the LUTs for sqrt and log
void box_muller(float u1, float u2, float& z0, float& z1, float log_lut[LOG_LUT_SIZE], float sqrt_lut[SQRT_LUT_SIZE]) {
    float r = sqrt_lut_lookup(-2.0f * log_lut_lookup(u1, log_lut), sqrt_lut);
    float theta = 2.0f * 3.14159f * u2;

    z0 = r * sin_lut(theta);
    z1 = r * cos_lut(theta);
}

// Top-level function to generate normally distributed random numbers
void generate_normal(float& z0, float& z1, ap_uint<LFSR_BITS>& lfsr, float log_lut[LOG_LUT_SIZE], float sqrt_lut[SQRT_LUT_SIZE]) {
    #pragma HLS pipeline
    #pragma HLS inline

    // Step 1: Generate two uniform random numbers between 0 and 1
    float u1 = uniform_random(lfsr);
    float u2 = uniform_random(lfsr);

    // Step 2: Apply Box-Muller Transform using LUTs to generate normal distribution
    box_muller(u1, u2, z0, z1, log_lut, sqrt_lut);
}

template<typename CONFIG_T>
void normaldist(
    float res[CONFIG_T::n_out]
) {
    #ifdef __HLS_SYN__
        bool initialized = false;
        float log_lut[LOG_LUT_SIZE];
        float sqrt_lut[SQRT_LUT_SIZE];
    #else
        static bool initialized = false;
        static float log_lut[LOG_LUT_SIZE];
        static float sqrt_lut[SQRT_LUT_SIZE];
    #endif
    if (!initialized) {
        init_log_lut(log_lut);
        init_sqrt_lut(sqrt_lut);
        initialized = true;
    }

    ap_uint<LFSR_BITS> lfsr = 0xACE1u;
    int i;
    for (i = 0; i < CONFIG_T::n_out - 1; i += 2) {
        generate_normal(res[i], res[i+1], lfsr, log_lut, sqrt_lut);
    }
    if (i < CONFIG_T::n_out) {
        float z1;
        generate_normal(res[i], z1, lfsr, log_lut, sqrt_lut);
    }
}

}

#endif