// Simple standalone HIP test to verify ROCm setup
// Tests: memory allocation, data transfer, kernel execution
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>

#define CHECK_HIP_ERROR(err) \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        exit(1); \
    }

// Simple kernel that adds 1 to each element
__global__ void add_one_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] + 1.0f;
    }
}

int main() {
    std::cout << "=== Simple HIP Test ===" << std::endl;

    // Check device count
    int device_count = 0;
    CHECK_HIP_ERROR(hipGetDeviceCount(&device_count));
    std::cout << "Found " << device_count << " HIP device(s)" << std::endl;

    if (device_count == 0) {
        std::cout << "❌ No HIP devices found!" << std::endl;
        return 1;
    }

    // Set device
    CHECK_HIP_ERROR(hipSetDevice(0));

    // Get device properties
    hipDeviceProp_t props;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&props, 0));
    std::cout << "Device: " << props.name << std::endl;
    std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;

    // Create test data
    const int size = 10;
    std::vector<float> h_input(size);
    std::vector<float> h_output(size);

    // Initialize input data
    for (int i = 0; i < size; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    std::cout << "Input data: ";
    for (int i = 0; i < size; ++i) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    // Allocate GPU memory
    float* d_data = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_data, size * sizeof(float)));
    std::cout << "✅ GPU memory allocated" << std::endl;

    // Copy input data to GPU
    CHECK_HIP_ERROR(hipMemcpy(d_data, h_input.data(), size * sizeof(float), hipMemcpyHostToDevice));
    std::cout << "✅ Data copied to GPU" << std::endl;

    // Launch kernel
    dim3 block_dim(256);
    dim3 grid_dim((size + 255) / 256);

    hipLaunchKernelGGL(add_one_kernel, grid_dim, block_dim, 0, 0, d_data, size);
    CHECK_HIP_ERROR(hipGetLastError());
    std::cout << "✅ Kernel launched successfully" << std::endl;

    // Wait for kernel to finish
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    std::cout << "✅ Kernel execution completed" << std::endl;

    // Copy result back to CPU
    CHECK_HIP_ERROR(hipMemcpy(h_output.data(), d_data, size * sizeof(float), hipMemcpyDeviceToHost));
    std::cout << "✅ Data copied back to CPU" << std::endl;

    // Verify results
    bool success = true;
    std::cout << "Output data: ";
    for (int i = 0; i < size; ++i) {
        std::cout << h_output[i] << " ";
        if (h_output[i] != h_input[i] + 1.0f) {
            success = false;
        }
    }
    std::cout << std::endl;

    // Free GPU memory
    CHECK_HIP_ERROR(hipFree(d_data));

    if (success) {
        std::cout << "✅ TEST PASSED! HIP is working correctly." << std::endl;
        std::cout << "GPU memory allocation, data transfer, and kernel execution all work." << std::endl;
        return 0;
    } else {
        std::cout << "❌ TEST FAILED! Results don't match expected values." << std::endl;
        return 1;
    }
}
