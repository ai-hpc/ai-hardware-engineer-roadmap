// jllm.h — Jetson LLM Runtime: master header
// Target: Orin Nano Super 8GB (SM 8.7, 102 GB/s LPDDR5, up to 67 sparse INT8 TOPS GPU; no DLA)

#pragma once

#include "jllm_memory.h"
#include "jllm_jetson.h"
#include "jllm_engine.h"
#include "jllm_kernels.h"

#define JLLM_VERSION_MAJOR 0
#define JLLM_VERSION_MINOR 1
#define JLLM_VERSION_PATCH 0

// SM 8.7 constants (Orin Nano/NX/AGX)
// JLLM_WARP_SIZE and JLLM_SHARED_MEM_SM defined in jllm_kernels.h
// Detected at runtime — these are defaults for Orin Nano 8GB (8 SMs in standard and Super modes)
// Query cudaGetDeviceProperties for the actual device and selected mode.
static constexpr int JLLM_SM_COUNT       = 8;   // override: cudaGetDeviceProperties
static constexpr int JLLM_CUDA_CORES     = 1024;  // SM_COUNT × 128
static constexpr int JLLM_TENSOR_CORES   = 32;    // SM_COUNT × 4
static constexpr int JLLM_MAX_THREADS_SM = 1536;
static constexpr int JLLM_MAX_WARPS_SM   = 48;
static constexpr int JLLM_REG_FILE_SM    = 256 * 1024;
static constexpr int JLLM_L2_CACHE       = 0; // unknown here; query l2CacheSize

static constexpr float JLLM_DRAM_BW_GBS  = 102.0f;
static constexpr float JLLM_GPU_TOPS_INT8 = 67.0f;
static constexpr float JLLM_DLA_TOPS_INT8 = 0.0f;
static constexpr float JLLM_RIDGE_POINT   = JLLM_GPU_TOPS_INT8 * 1000.0f / JLLM_DRAM_BW_GBS;

// Ridge value is sparse INT8 operations/byte at advertised peaks, not a
// measured dense/FP16/LLM threshold. TOPS/GB/s needs the factor of 1000.
