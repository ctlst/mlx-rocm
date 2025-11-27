// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/jit_module.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/version.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <fmt/format.h>
#include <hip/hiprtc.h>

namespace mlx::core::rocm {

namespace {

#define CHECK_HIPRTC_ERROR(cmd) check_hiprtc_error(#cmd, (cmd))

void check_hiprtc_error(const char* name, hiprtcResult err) {
  if (err != HIPRTC_SUCCESS) {
    throw std::runtime_error(
        fmt::format("{} failed: {}", name, hiprtcGetErrorString(err)));
  }
}

// Return the location of the ROCm installation.
const std::string& rocm_home() {
  static std::string home = []() -> std::string {
    const char* home = std::getenv("ROCM_PATH");
    if (home) {
      return home;
    }
    // Default paths
#if defined(__linux__)
    home = "/opt/rocm";
    if (std::filesystem::exists(home)) {
      return home;
    }
#endif
    throw std::runtime_error("Environment variable ROCM_PATH is not set.");
  }();
  return home;
}

// Get the cache directory for storing compiled results.
const std::filesystem::path& hip_cache_dir() {
  static std::filesystem::path cache = []() -> std::filesystem::path {
    std::filesystem::path cache;
    if (auto c = std::getenv("MLX_HIP_CACHE_DIR"); c) {
      cache = c;
    } else {
      cache =
          std::filesystem::temp_directory_path() / "mlx" / version() / "hip";
    }
    if (!std::filesystem::exists(cache)) {
      std::error_code error;
      if (!std::filesystem::create_directories(cache, error)) {
        return std::filesystem::path();
      }
    }
    return cache;
  }();
  return cache;
}

} // namespace

std::filesystem::path get_hip_path(
    const std::filesystem::path& cache_dir,
    const std::string& module_name) {
#ifdef _WIN32
  constexpr int max_file_name_length = 140;
#else
  constexpr int max_file_name_length = 245;
#endif

  if (module_name.size() <= max_file_name_length) {
    return cache_dir / (module_name + ".hip");
  }

  auto hip_path = cache_dir;
  int offset = 0;
  while (module_name.size() - offset > max_file_name_length) {
    hip_path /= module_name.substr(offset, max_file_name_length);
    offset += max_file_name_length;
  }
  hip_path /= module_name.substr(offset) + ".hip";

  return hip_path;
}

// Try to read the cached |hip_code| and |hip_kernels| from |cache_dir|.
bool read_cached_hip(
    const std::filesystem::path& cache_dir,
    const std::string& module_name,
    std::string& hip_code,
    std::vector<std::pair<std::string, std::string>>& hip_kernels) {
  if (cache_dir.empty()) {
    return false;
  }

  auto hip_path = get_hip_path(cache_dir, module_name);
  std::error_code error;
  auto hip_size = std::filesystem::file_size(hip_path, error);
  if (error) {
    return false;
  }
  std::ifstream hip_file(hip_path, std::ios::binary);
  if (!hip_file.good()) {
    return false;
  }
  hip_code.resize(hip_size);
  hip_file.read(hip_code.data(), hip_size);

  std::ifstream txt_file(hip_path.replace_extension(".txt"), std::ios::binary);
  std::string line;
  while (std::getline(txt_file, line)) {
    auto tab = line.find('\t');
    if (tab != std::string::npos) {
      hip_kernels.emplace_back(line.substr(0, tab), line.substr(tab + 1));
    }
  }
  return true;
}

// Write the |hip_code| and |hip_kernels| to |cache_dir| with |name|.
void write_cached_hip(
    const std::filesystem::path& cache_dir,
    const std::string& module_name,
    const std::string& hip_code,
    const std::vector<std::pair<std::string, std::string>>& hip_kernels,
    const std::string& source_code) {
  if (cache_dir.empty()) {
    return;
  }

  auto hip_path = get_hip_path(cache_dir, module_name);

  // Ensure that the directory exists
  auto parent = hip_path.parent_path();
  if (parent != cache_dir) {
    std::filesystem::create_directories(parent);
  }

  // Write the compiled code and mangled names
  std::ofstream hip_file(hip_path, std::ios::binary);
  if (!hip_code.empty()) {
    hip_file.write(&hip_code.front(), hip_code.size());
  }
  std::ofstream txt_file(hip_path.replace_extension(".txt"), std::ios::binary);
  for (const auto& [name, mangled] : hip_kernels) {
    txt_file << name << "\t" << mangled << std::endl;
  }

  // Write the generated code
  std::ofstream source_file(hip_path.replace_extension(".hip"));
  source_file << source_code;
}

void compile(
    Device& device,
    const std::string& module_name,
    const std::string& source,
    const std::vector<std::string>& kernel_names,
    std::string& hip_code,
    std::vector<std::pair<std::string, std::string>>& hip_kernels) {
  // Create the program
  hiprtcProgram prog;
  CHECK_HIPRTC_ERROR(hiprtcCreateProgram(
      &prog,
      source.c_str(),
      (module_name + ".hip").c_str(),
      0,     // numHeaders
      nullptr, // headers
      nullptr  // includeNames
  ));
  std::unique_ptr<hiprtcProgram, void (*)(hiprtcProgram*)> prog_freer(
      &prog,
      [](hiprtcProgram* p) { CHECK_HIPRTC_ERROR(hiprtcDestroyProgram(p)); });

  // Add name expressions for kernels
  for (const auto& name : kernel_names) {
    CHECK_HIPRTC_ERROR(hiprtcAddNameExpression(prog, name.c_str()));
  }

  // Compile program
  std::vector<const char*> args;

  // Set target architecture
  std::string arch = fmt::format("--gpu-architecture=gfx{}", device.compute_capability_major() * 100 + device.compute_capability_minor());
  args.push_back(arch.c_str());

  // Include paths
  std::string rocm_include = fmt::format("--include-path={}/include", rocm_home());
  args.push_back(rocm_include.c_str());

  hiprtcResult compile_result = hiprtcCompileProgram(prog, args.size(), args.data());

  // Get compilation log
  size_t log_size;
  CHECK_HIPRTC_ERROR(hiprtcGetProgramLogSize(prog, &log_size));
  if (log_size > 1) {
    std::string log;
    log.resize(log_size);
    CHECK_HIPRTC_ERROR(hiprtcGetProgramLog(prog, &log[0]));
    if (compile_result != HIPRTC_SUCCESS) {
      throw std::runtime_error(fmt::format(
          "hiprtc compilation failed: {}\nLog: {}", hiprtcGetErrorString(compile_result), log));
    }
  }

  if (compile_result != HIPRTC_SUCCESS) {
    throw std::runtime_error(fmt::format(
        "hiprtc compilation failed: {}", hiprtcGetErrorString(compile_result)));
  }

  // Get lowered names
  for (const auto& name : kernel_names) {
    const char* lowered_name;
    CHECK_HIPRTC_ERROR(hiprtcGetLoweredName(prog, name.c_str(), &lowered_name));
    hip_kernels.emplace_back(name, lowered_name);
  }

  // Get the compiled code
  size_t code_size;
  CHECK_HIPRTC_ERROR(hiprtcGetCodeSize(prog, &code_size));
  hip_code.resize(code_size);
  CHECK_HIPRTC_ERROR(hiprtcGetCode(prog, hip_code.data()));
}

JitModule::JitModule(
    Device& device,
    const std::string& module_name,
    const KernelBuilder& builder,
    bool cache)
    : module_(nullptr) {
  auto [precompiled, source, kernel_names] = builder();

  std::string hip_code;
  std::vector<std::pair<std::string, std::string>> hip_kernels;

  if (cache) {
    auto cache_dir = hip_cache_dir();
    if (read_cached_hip(cache_dir, module_name, hip_code, hip_kernels)) {
      // Verify that we have all the expected kernels
      bool all_found = true;
      for (const auto& expected : kernel_names) {
        bool found = false;
        for (const auto& [name, _] : hip_kernels) {
          if (name == expected) {
            found = true;
            break;
          }
        }
        if (!found) {
          all_found = false;
          break;
        }
      }
      if (!all_found) {
        hip_code.clear();
        hip_kernels.clear();
      }
    }
  }

  if (hip_code.empty()) {
    compile(device, module_name, source, kernel_names, hip_code, hip_kernels);
    if (cache) {
      auto cache_dir = hip_cache_dir();
      write_cached_hip(cache_dir, module_name, hip_code, hip_kernels, source);
    }
  }

  // Load the module
  CHECK_HIP_ERROR(hipModuleLoadData(&module_, hip_code.c_str()));

  // Extract kernel functions
  for (const auto& [name, mangled] : hip_kernels) {
    hipFunction_t kernel;
    CHECK_HIP_ERROR(hipModuleGetFunction(&kernel, module_, mangled.c_str()));

    // For ROCm, we'll assume 1D block for now
    uint block_size = 256;
    kernels_[name] = std::make_tuple(kernel, true, block_size);
  }
}

JitModule::~JitModule() {
  if (module_) {
    CHECK_HIP_ERROR(hipModuleUnload(module_));
  }
}

hipFunction_t JitModule::get_kernel(
    const std::string& kernel_name,
    std::function<void(hipFunction_t)> configure_kernel) {
  auto it = kernels_.find(kernel_name);
  if (it == kernels_.end()) {
    throw std::runtime_error(fmt::format("Kernel '{}' not found", kernel_name));
  }

  auto [kernel, configured, block_size] = it->second;
  if (!configured && configure_kernel) {
    configure_kernel(kernel);
    std::get<1>(it->second) = true;
  }

  return kernel;
}

std::pair<hipFunction_t, uint> JitModule::get_kernel_and_dims(
    const std::string& kernel_name,
    std::function<void(hipFunction_t)> configure_kernel) {
  auto kernel = get_kernel(kernel_name, configure_kernel);
  auto it = kernels_.find(kernel_name);
  auto [_, __, block_size] = it->second;
  return {kernel, block_size};
}

std::unordered_map<std::string, JitModule>& get_jit_module_cache() {
  static std::unordered_map<std::string, JitModule> cache;
  return cache;
}

JitModule& get_jit_module(
    const mlx::core::Device& device,
    const std::string& name,
    const KernelBuilder& builder,
    bool use_disk_cache) {
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);

  auto& cache = get_jit_module_cache();
  auto it = cache.find(name);
  if (it == cache.end()) {
    // Get the underlying rocm device
    auto& rocm_device = rocm::device(device);
    it = cache.try_emplace(name, rocm_device, name, builder, use_disk_cache).first;
  }
  return it->second;
}

} // namespace mlx::core::rocm
