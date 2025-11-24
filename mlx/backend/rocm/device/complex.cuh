// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

template <typename T>
struct complex_t {
  T real_;
  T imag_;

  __host__ __device__ complex_t() : real_(0), imag_(0) {}
  __host__ __device__ complex_t(T r) : real_(r), imag_(0) {}
  __host__ __device__ complex_t(T r, T i) : real_(r), imag_(i) {}

  __host__ __device__ T real() const { return real_; }
  __host__ __device__ T imag() const { return imag_; }

  __host__ __device__ complex_t operator+(const complex_t& other) const {
    return complex_t(real_ + other.real_, imag_ + other.imag_);
  }

  __host__ __device__ complex_t operator-(const complex_t& other) const {
    return complex_t(real_ - other.real_, imag_ - other.imag_);
  }

  __host__ __device__ complex_t operator*(const complex_t& other) const {
    return complex_t(
        real_ * other.real_ - imag_ * other.imag_,
        real_ * other.imag_ + imag_ * other.real_);
  }

  __host__ __device__ complex_t operator/(const complex_t& other) const {
    T denom = other.real_ * other.real_ + other.imag_ * other.imag_;
    return complex_t(
        (real_ * other.real_ + imag_ * other.imag_) / denom,
        (imag_ * other.real_ - real_ * other.imag_) / denom);
  }

  __host__ __device__ bool operator==(const complex_t& other) const {
    return real_ == other.real_ && imag_ == other.imag_;
  }

  __host__ __device__ bool operator!=(const complex_t& other) const {
    return !(*this == other);
  }
};

template <typename T>
__host__ __device__ complex_t<T> conj(const complex_t<T>& z) {
  return complex_t<T>(z.real(), -z.imag());
}

template <typename T>
__host__ __device__ T abs(const complex_t<T>& z) {
  return sqrt(z.real() * z.real() + z.imag() * z.imag());
}

template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<complex_t<T>> : std::true_type {};

template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;

using complex64_t = complex_t<float>;

} // namespace mlx::core::rocm

