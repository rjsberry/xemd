// xemd, Copyright (c) 2018, Richard Berry <rjsberry@protonmail.com>
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.

#ifndef INCLUDE_XEMD_EMPIRICAL_MODE_DECOMPOSITION_TEMPLATE_LIBRARY_HPP_
#define INCLUDE_XEMD_EMPIRICAL_MODE_DECOMPOSITION_TEMPLATE_LIBRARY_HPP_

#include <cmath>
#include <iostream>
#include <type_traits>

#if defined(XEMD_USE_XTENSOR_JULIA)
  #include <xtensor-julia/jltensor.hpp>
  #include <xtensor-julia/jlarray.hpp>

  namespace xemd {
  namespace array_type {
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using tensor = xt::jltensor<T, 1>;
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using array = xt::jlarray<T>;
  }  // namespace array_type
  }  // namespace xemd

#elif defined(XEMD_USE_XTENSOR_PYTHON)
  #include <xtensor-python/pytensor.hpp>
  #include <xtensor-python/pyarray.hpp>

  namespace xemd {
  namespace array_type {
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using tensor = xt::pytensor<T, 1>;
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using array = xt::pyarray<T>;
  }  // namespace array_type
  }  // namespace xemd

#elif defined(XEMD_USE_XTENSOR_R)
  #include <xtensor-r/rtensor.hpp>
  #include <xtensor-r/rarray.hpp>

  namespace xemd {
  namespace array_type {
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using tensor = xt::rtensor<T, 1>;
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using array = xt::rarray<T>;
  }  // namespace array_type
  }  // namespace xemd

#else
  #include <xtensor/xtensor.hpp>
  #include <xtensor/xarray.hpp>

  namespace xemd {
  namespace array_type {
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using tensor = xt::xtensor<T, 1>;
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using array = xt::xarray<T>;
  }  // namespace array_type
  }  // namespace xemd

#endif

namespace xemd {

namespace xutils {

template<typename T> inline
xemd::array_type::tensor<T>
Diff(const xemd::array_type::tensor<T>& x) {
  xemd::array_type::tensor<T> d = xt::zeros<T>({x.size() - 1});
  for (std::size_t i = 0; i < x.size() - 1; ++i) {
    d[i] = x[i + 1] - x[i];
  }
  return d;
}

template<typename T> inline
bool
IsMonotonic(const xemd::array_type::tensor<T>& x) {
  if (xt::all(Diff<T>(x) >= 0) || xt::all(Diff<T>(x) <= 0)) {
    return true;
  }
  return false;
}

template<typename T> inline
std::size_t
NumImfs(const xemd::array_type::tensor<T>& x) {
  auto N = x.size();
  if (N <= 3) {
    return 1;
  } else {
    return std::floor(std::log2(N));
  }
}

}  // namespace xutils

template<typename T>
void
emd(const xemd::array_type::tensor<T>& xin) {
  std::cout << "CORE: `xemd::emd` called" << std::endl;
}

template<typename T>
void
eemd(const xemd::array_type::tensor<T>& xin) {
  std::cout << "CORE: `xemd::eemd` called" << std::endl;
}

template<typename T>
void
ceemdan(const xemd::array_type::tensor<T>& xin) {
  std::cout << "CORE: `xemd::ceemdan` called" << std::endl;
}

}  // namespace xemd

#endif  // INCLUDE_XEMD_EMPIRICAL_MODE_DECOMPOSITION_TEMPLATE_LIBRARY_HPP_
