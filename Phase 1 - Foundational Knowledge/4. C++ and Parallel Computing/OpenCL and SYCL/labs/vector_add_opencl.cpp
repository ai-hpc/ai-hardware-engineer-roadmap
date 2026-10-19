// OpenCL 1.2 API baseline; CPU device preferred, any available device as fallback.
// g++ -std=c++17 -O2 vector_add_opencl.cpp -lOpenCL -o vector_add_opencl
// ./vector_add_opencl [length=1025 | --self-test | --inject-build-error]
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

void check(cl_int status, const char* operation) {
    if (status != CL_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed with OpenCL status " + std::to_string(status));
}
template<class T, auto Release> struct Handle {
    T value = nullptr;
    explicit Handle(T object) : value(object) {}
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    ~Handle() { if (value) Release(value); }
};
cl_device_id select_device() {
    cl_uint count = 0;
    check(clGetPlatformIDs(0, nullptr, &count), "enumerate platforms");
    if (count == 0) throw std::runtime_error("no OpenCL platform: an ICD loader needs a device implementation");
    std::vector<cl_platform_id> platforms(count);
    check(clGetPlatformIDs(count, platforms.data(), nullptr), "get platforms");
    const cl_device_type types[] = {CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_ALL};
    for (cl_device_type type : types) {
        for (auto platform : platforms) {
            cl_device_id device;
            const auto status = clGetDeviceIDs(platform, type, 1, &device, nullptr);
            if (status == CL_SUCCESS) return device;
            if (status != CL_DEVICE_NOT_FOUND) check(status, "enumerate devices");
        }
    }
    throw std::runtime_error("no OpenCL device found");
}
std::string device_info(cl_device_id device, cl_device_info field) {
    std::size_t length = 0;
    check(clGetDeviceInfo(device, field, 0, nullptr, &length), "device info size");
    std::vector<char> data(length);
    check(clGetDeviceInfo(device, field, length, data.data(), nullptr), "device info");
    return std::string(data.data());
}
void verify_length(cl_context context, cl_command_queue queue, cl_kernel kernel,
                   cl_device_id device, int n) {
    if (n == 0) {
        std::puts("PASS n=0 empty result (no zero-sized buffer or kernel launch)");
        return;
    }
    std::vector<float> a(n), b(n), c(n, -9999.0f);
    for (int i = 0; i < n; ++i) { a[i] = static_cast<float>(i % 31); b[i] = -static_cast<float>(i % 19); }
    const std::size_t bytes = a.size() * sizeof(float);
    cl_int status;
    Handle<cl_mem, clReleaseMemObject> da(clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &status));
    check(status, "allocate input A");
    Handle<cl_mem, clReleaseMemObject> db(clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &status));
    check(status, "allocate input B");
    Handle<cl_mem, clReleaseMemObject> dc(clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &status));
    check(status, "allocate output");
    check(clEnqueueWriteBuffer(queue, da.value, CL_TRUE, 0, bytes, a.data(), 0, nullptr, nullptr), "write A");
    check(clEnqueueWriteBuffer(queue, db.value, CL_TRUE, 0, bytes, b.data(), 0, nullptr, nullptr), "write B");
    check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &da.value), "set A");
    check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &db.value), "set B");
    check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &dc.value), "set C");
    check(clSetKernelArg(kernel, 3, sizeof(n), &n), "set length");
    // Choose a supported one-dimensional group; padding tests the i < n guard.
    std::size_t kernel_max = 0;
    check(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(kernel_max), &kernel_max, nullptr), "kernel group size");
    cl_uint dimensions = 0;
    check(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(dimensions), &dimensions, nullptr), "work-item dimensions");
    std::vector<std::size_t> work_item_max(dimensions);
    check(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, dimensions * sizeof(std::size_t), work_item_max.data(), nullptr), "work-item limits");
    const std::size_t local = std::min({std::size_t(64), kernel_max, work_item_max.at(0)});
    if (local == 0) throw std::runtime_error("device reported zero work-group capacity");
    const std::size_t global = (a.size() + local - 1) / local * local;
    cl_event raw_event;
    check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, &raw_event), "enqueue kernel");
    Handle<cl_event, clReleaseEvent> event(raw_event);
    // Queue is in order. Blocking read also observes preceding kernel completion.
    check(clEnqueueReadBuffer(queue, dc.value, CL_TRUE, 0, bytes, c.data(), 0, nullptr, nullptr), "read result");
    for (int i = 0; i < n; ++i)
        if (c[i] != a[i] + b[i]) throw std::runtime_error("scalar-reference mismatch at element " + std::to_string(i));
    cl_ulong start, end;
    check(clGetEventProfilingInfo(event.value, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr), "profile start");
    check(clGetEventProfilingInfo(event.value, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr), "profile end");
    if (end < start) throw std::runtime_error("invalid device profiling timestamps");
    std::printf("PASS n=%d global=%zu local=%zu kernel_event_ns=%llu\n", n, global, local, static_cast<unsigned long long>(end - start));
}
int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("too many arguments");
        const std::string_view argument = argc == 2 ? argv[1] : "1025";
        const bool self_test = argument == "--self-test";
        const bool inject = argument == "--inject-build-error";
        int n = 1025;
        if (!self_test && !inject) {
            auto parsed = std::from_chars(argument.data(), argument.data() + argument.size(), n);
            if (parsed.ec != std::errc{} || parsed.ptr != argument.data() + argument.size() || n < 0 || n > 1000000)
                throw std::invalid_argument("length must be an integer in [0,1000000]");
        }
        auto device = select_device();
        std::printf("device=%s\nversion=%s\ndriver=%s\n", device_info(device, CL_DEVICE_NAME).c_str(),
                    device_info(device, CL_DEVICE_VERSION).c_str(), device_info(device, CL_DRIVER_VERSION).c_str());
        cl_int status;
        Handle<cl_context, clReleaseContext> context(clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status));
        check(status, "create context");
        Handle<cl_command_queue, clReleaseCommandQueue> queue(clCreateCommandQueue(context.value, device, CL_QUEUE_PROFILING_ENABLE, &status));
        check(status, "create queue");
        const char* source = inject ? "__kernel void vector_add( { COURSE_INTENTIONAL_SYNTAX_ERROR" : R"CLC(
            __kernel void vector_add(__global const float* a, __global const float* b,
                                     __global float* c, const int n) {
                size_t i = get_global_id(0);
                if (i < (size_t)n) c[i] = a[i] + b[i];
            }
        )CLC";
        Handle<cl_program, clReleaseProgram> program(clCreateProgramWithSource(context.value, 1, &source, nullptr, &status));
        check(status, "create program");
        status = clBuildProgram(program.value, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
        if (status != CL_SUCCESS) {
            std::size_t length = 0;
            check(clGetProgramBuildInfo(program.value, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length), "build log size");
            std::vector<char> log(length + 1, '\0');
            check(clGetProgramBuildInfo(program.value, device, CL_PROGRAM_BUILD_LOG, length, log.data(), nullptr), "build log");
            std::fprintf(stderr, "BUILD_LOG:\n%s\n", log.data());
            check(status, "build program");
        }
        if (inject) throw std::runtime_error("intentional invalid kernel unexpectedly compiled");
        Handle<cl_kernel, clReleaseKernel> kernel(clCreateKernel(program.value, "vector_add", &status));
        check(status, "create kernel");
        if (self_test) for (int length : {0, 1, 7, 63, 64, 65, 255, 256, 257, 1025})
            verify_length(context.value, queue.value, kernel.value, device, length);
        else verify_length(context.value, queue.value, kernel.value, device, n);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
