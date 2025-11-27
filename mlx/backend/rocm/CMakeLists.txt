# Filename rules in rocm backend:
#
# * Use .hip/.hpp if code contains device code, and .cpp/.h if not.
# * Device-only code should be put in device/ subdir.
# * Files in device/ subdir should not include files outside.
target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/rocm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/worker.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/jit_module.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_contiguous.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/arange.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/binary_two.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/ternary.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/all_reduce.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/col_reduce.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/row_reduce.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/init_reduce.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/softmax.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/random.hip
          ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/rocblas_gemm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/kernel_utils.hip)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/binary)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/unary)

target_compile_definitions(mlx PRIVATE MLX_USE_ROCM)

# Compute capability for gfx1030 (RDNA 2)
if(NOT DEFINED MLX_ROCM_ARCHITECTURES)
  set(MLX_ROCM_ARCHITECTURES "gfx1030")
endif()
message(STATUS "ROCm architectures: ${MLX_ROCM_ARCHITECTURES}")

# Set HIP architectures for the target
set_target_properties(mlx PROPERTIES HIP_ARCHITECTURES ${MLX_ROCM_ARCHITECTURES})

# For cross-compilation, rely on environment variables instead of setting HIP_ARCHITECTURES
# which would apply to all files in the target

# Make HIP runtime APIs available
target_include_directories(mlx PRIVATE ${hip_INCLUDE_DIRS})

# Link with HIP runtime
target_link_libraries(mlx PRIVATE hip::device)

# Use rocBLAS
target_link_libraries(mlx PRIVATE roc::rocblas)

# Use hipRTC for JIT compilation
target_link_libraries(mlx PRIVATE hiprtc::hiprtc)

# Link roctracer if available (for profiling)
find_library(ROCTX_LIBRARY NAMES roctx64 PATHS ${ROCM_PATH}/lib /opt/rocm/lib)
if(ROCTX_LIBRARY)
  message(STATUS "Found roctracer: ${ROCTX_LIBRARY}")
  target_link_libraries(mlx PRIVATE ${ROCTX_LIBRARY})
else()
  message(STATUS "roctracer not found, profiling markers disabled")
endif()

# Enable extended lambda for device code
target_compile_options(mlx PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-allow-device-init>)

