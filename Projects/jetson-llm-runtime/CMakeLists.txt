cmake_minimum_required(VERSION 3.20)
project(jetson-llm LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(JLLM_BUILD_JETSON "Build the experimental Jetson CUDA runtime" OFF)
option(JLLM_SANITIZE "Enable ASan/UBSan for the CPU teaching components" OFF)

add_library(jllm_cpu STATIC src/engine/gguf.cpp src/engine/sample.cpp src/engine/tokenizer.cpp src/engine/model_config.cpp src/memory/budget.cpp)
target_include_directories(jllm_cpu PUBLIC include)
target_compile_options(jllm_cpu PRIVATE -Wall -Wextra -Wpedantic)
if(JLLM_SANITIZE)
    target_compile_options(jllm_cpu PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(jllm_cpu PUBLIC -fsanitize=address,undefined)
endif()
add_executable(gguf-inspect src/gguf_inspect.cpp)
target_link_libraries(gguf-inspect PRIVATE jllm_cpu)
add_executable(test_sampling tests/test_sampling.cpp)
target_link_libraries(test_sampling PRIVATE jllm_cpu)
add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE jllm_cpu)
add_executable(test_memory_contracts tests/test_memory_contracts.cpp)
target_link_libraries(test_memory_contracts PRIVATE jllm_cpu)
# Compile the actual pool control logic against a bounded host-only CUDA shim.
# This checks arithmetic/lifetime paths, not CUDA allocation or GPU behavior.
add_executable(test_allocator_contracts tests/test_allocator_contracts.cpp src/memory/pool.cpp src/memory/kv_cache.cpp)
target_include_directories(test_allocator_contracts BEFORE PRIVATE tests/cuda_stub)
target_link_libraries(test_allocator_contracts PRIVATE jllm_cpu)
add_executable(test_attention_reference tests/test_attention_reference.cpp src/memory/kv_cache.cpp)
target_include_directories(test_attention_reference BEFORE PRIVATE tests/cuda_stub)
target_link_libraries(test_attention_reference PRIVATE jllm_cpu)
add_executable(test_decode_scratch tests/test_decode_scratch.cpp src/memory/pool.cpp)
target_include_directories(test_decode_scratch BEFORE PRIVATE tests/cuda_stub)
target_link_libraries(test_decode_scratch PRIVATE jllm_cpu)
add_executable(benchmark_cpu src/benchmark_cpu.cpp)
target_link_libraries(benchmark_cpu PRIVATE jllm_cpu)
enable_testing()
add_test(NAME sampling COMMAND test_sampling)
add_test(NAME memory_contracts COMMAND test_memory_contracts)
add_test(NAME allocator_cpu_shim COMMAND test_allocator_contracts)
add_test(NAME attention_cpu_oracle COMMAND test_attention_reference)
add_test(NAME decode_scratch_cpu_contract COMMAND test_decode_scratch)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME gguf_tokenizer_contracts COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/labs/test_cpu_runtime.py)
    set_tests_properties(gguf_tokenizer_contracts PROPERTIES ENVIRONMENT "JLLM_CPU_BUILD=${CMAKE_CURRENT_BINARY_DIR}")
endif()
if(NOT JLLM_BUILD_JETSON)
    return()
endif()
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    message(FATAL_ERROR "JLLM_BUILD_JETSON needs a Jetson/aarch64 toolchain; CPU lessons do not.")
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 87)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# ── Sources ──────────────────────────────────────────────────────────────
set(MEMORY_SRC
    src/memory/kv_cache.cpp
    src/memory/pool.cpp
)

set(JETSON_SRC
    src/jetson/power.cpp
    src/jetson/thermal.cpp
    src/jetson/sysinfo.cpp
)

set(KERNEL_SRC
    src/kernels/gemv_q4.cu
    src/kernels/fused_norm.cu
    src/kernels/attention.cu
    src/kernels/rope.cu
    src/kernels/softmax.cu
    src/kernels/convert.cu
)

set(ENGINE_SRC
    src/engine/model.cpp
    src/engine/decode.cu
)

set(SERVER_SRC
    src/server/http_server.cpp
)

# ── Library ──────────────────────────────────────────────────────────────
add_library(jetson_llm_core STATIC
    ${MEMORY_SRC} ${JETSON_SRC} ${KERNEL_SRC} ${ENGINE_SRC}
)
target_include_directories(jetson_llm_core PUBLIC include)
target_link_libraries(jetson_llm_core PUBLIC
    jllm_cpu CUDA::cudart CUDA::cublas Threads::Threads
)

# ── CLI binary ───────────────────────────────────────────────────────────
add_executable(jetson-llm src/main.cpp)
target_link_libraries(jetson-llm PRIVATE jetson_llm_core)

# ── Server binary ────────────────────────────────────────────────────────
add_executable(jetson-llm-server src/main_server.cpp ${SERVER_SRC})
target_link_libraries(jetson-llm-server PRIVATE jetson_llm_core)

# ── Tests ────────────────────────────────────────────────────────────────
add_executable(test_memory tests/test_memory.cpp)
target_link_libraries(test_memory PRIVATE jetson_llm_core)
add_test(NAME memory COMMAND test_memory)

add_executable(test_kernels tests/test_kernels.cu)
target_link_libraries(test_kernels PRIVATE jetson_llm_core)
add_test(NAME kernels COMMAND test_kernels)
add_executable(test_attention tests/test_attention.cu)
target_link_libraries(test_attention PRIVATE jetson_llm_core)
add_test(NAME attention_cuda_parity COMMAND test_attention)
set_tests_properties(attention_cuda_parity PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(kernels memory PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_model_load tests/test_model_load.cpp)
target_link_libraries(test_model_load PRIVATE jetson_llm_core)
# Run with: ./build/test_model_load model.gguf

# Keep legacy assert-based GPU checks active in Release configurations.
foreach(test_target test_memory test_kernels test_model_load)
    target_compile_options(${test_target} PRIVATE -UNDEBUG)
endforeach()
