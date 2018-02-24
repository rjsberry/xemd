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

#include <iostream>
#include <type_traits>

#if defined(XEMD_USE_XTENSOR_JULIA)

  #include <xtensor-julia/jltensor.hpp>
  namespace {
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using _xtensor = xt::jltensor<T, 1>;
  }  // namespace
 
  #include <xtensor-julia/jlarray.hpp>
  namespace {
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using _xarray = xt::jlarray<T>;
  }  // namespace

#elif defined(XEMD_USE_XTENSOR_PYTHON)

  #include <xtensor-python/pytensor.hpp>
  namespace {
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using _xtensor = xt::pytensor<T, 1>;
  }  // namespace
 
  #include <xtensor-python/pyarray.hpp>
  namespace {
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using _xarray = xt::pyarray<T>;
  }  // namespace

#elif defined(XEMD_USE_XTENSOR_R)

  #include <xtensor-r/rtensor.hpp>
  namespace {
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using _xtensor = xt::rtensor<T, 1>;
  }  // namespace
 
  #include <xtensor-r/rarray.hpp>
  namespace {
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using _xarray = xt::rarray<T>;
  }  // namespace

#else

  #include <xtensor/xtensor.hpp>
  namespace {
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using _xtensor = xt::xtensor<T, 1>;
  }  // namespace
 
  #include <xtensor/xarray.hpp>
  namespace {
  template<
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  >
  using _xarray = xt::xarray<T>;
  }  // namespace

#endif

namespace xemd {

template<typename T>
void
emd(const _xtensor<T>& xin) {
  std::cout << "CORE: `xemd::emd` called" << std::endl;
}

template<typename T>
void
eemd(const _xtensor<T>& xin) {
  std::cout << "CORE: `xemd::eemd` called" << std::endl;
}

template<typename T>
void
ceemdan(const _xtensor<T>& xin) {
  std::cout << "CORE: `xemd::ceemdan` called" << std::endl;
}

}  // namespace xemd

#endif  // INCLUDE_XEMD_EMPIRICAL_MODE_DECOMPOSITION_TEMPLATE_LIBRARY_HPP_
