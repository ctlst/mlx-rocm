// Simple HIP test program to verify ROCm JIT compilation
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

const char* kernel_source = R"(
extern "C" {

__global__ void simple_add_kernel(float* output, float a, float b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a + b;
    }
}

}
)";

int main() {
    std::cout << "Testing ROCm hipRTC compilation..." << std::endl;

    // Check if HIP is available
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        std::cout << "❌ No HIP devices found: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "✅ Found " << device_count << " HIP device(s)" << std::endl;

    // Create hiprtc program
    hiprtcProgram prog;
    hiprtcResult result = hiprtcCreateProgram(&prog, kernel_source, "simple_kernel.hip", 0, nullptr, nullptr);
    if (result != HIPRTC_SUCCESS) {
        std::cout << "❌ Failed to create program: " << hiprtcGetErrorString(result) << std::endl;
        return 1;
    }
    std::cout << "✅ Created hipRTC program" << std::endl;

    // Compile the program
    const char* options[] = {"--gpu-architecture=gfx1030"};
    result = hiprtcCompileProgram(prog, 1, options);
    if (result != HIPRTC_SUCCESS) {
        // Get compilation log
        size_t log_size;
        hiprtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        hiprtcGetProgramLog(prog, &log[0]);
        std::cout << "❌ Compilation failed: " << hiprtcGetErrorString(result) << std::endl;
        std::cout << "Log: " << log << std::endl;
        hiprtcDestroyProgram(&prog);
        return 1;
    }
    std::cout << "✅ Program compiled successfully" << std::endl;

    // Get the compiled code
    size_t code_size;
    hiprtcGetCodeSize(prog, &code_size);
    std::vector<char> code(code_size);
    hiprtcGetCode(prog, code.data());

    // Load the module
    hipModule_t module;
    err = hipModuleLoadData(&module, code.data());
    if (err != hipSuccess) {
        std::cout << "❌ Failed to load module: " << hipGetErrorString(err) << std::endl;
        hiprtcDestroyProgram(&prog);
        return 1;
    }
    std::cout << "✅ Module loaded" << std::endl;

    // Get kernel function
    hipFunction_t kernel;
    err = hipModuleGetFunction(&kernel, module, "simple_add_kernel");
    if (err != hipSuccess) {
        std::cout << "❌ Failed to get kernel function: " << hipGetErrorString(err) << std::endl;
        hipModuleUnload(module);
        hiprtcDestroyProgram(&prog);
        return 1;
    }
    std::cout << "✅ Got kernel function" << std::endl;

    // Allocate host and device memory
    const int size = 10;
    std::vector<float> h_output(size);
    float* d_output;
    hipMalloc(&d_output, size * sizeof(float));

    // Launch kernel
    dim3 block_dim(256);
    dim3 grid_dim((size + 255) / 256);
    float a = 3.0f, b = 7.0f;
    void* args[] = {&d_output, &a, &b, &size};

    err = hipModuleLaunchKernel(kernel, grid_dim.x, grid_dim.y, grid_dim.z,
                               block_dim.x, block_dim.y, block_dim.z,
                               0, 0, args, nullptr);
    if (err != hipSuccess) {
        std::cout << "❌ Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_output);
        hipModuleUnload(module);
        hiprtcDestroyProgram(&prog);
        return 1;
    }
    std::cout << "✅ Kernel launched successfully" << std::endl;

    // Copy result back
    hipMemcpy(h_output.data(), d_output, size * sizeof(float), hipMemcpyDeviceToHost);

    // Check result
    bool success = true;
    for (int i = 0; i < size; ++i) {
        if (h_output[i] != 10.0f) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "✅ Test PASSED! Kernel computed " << h_output[0] << " (expected 10.0)" << std::endl;
    } else {
        std::cout << "❌ Test FAILED! Got " << h_output[0] << " (expected 10.0)" << std::endl;
    }

    // Cleanup
    hipFree(d_output);
    hipModuleUnload(module);
    hiprtcDestroyProgram(&prog);

    return success ? 0 : 1;
}
