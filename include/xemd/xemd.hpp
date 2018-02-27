#ifndef INCLUDE_XEMD_EMPIRICAL_MODE_DECOMPOSITION_TEMPLATE_LIBRARY_HPP_
#define INCLUDE_XEMD_EMPIRICAL_MODE_DECOMPOSITION_TEMPLATE_LIBRARY_HPP_

// xemd: https://github.com/rjsberry/xemd
//
// Copyright (C) 2018, Richard Berry <rjsberry@protonmail.com>
//
// Distributed under the terms of BSD 2-Clause "simplified" license. (See
// accompanying file LICENSE, or copy at
// https://github.com/rjsberry/xemd/blob/master/LICENSE)
//

#include <cassert>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

#include <xtensor/xadapt.hpp>

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
T
Extrapolate(T x0, T y0, T x1, T y1, T x) {
  return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
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

namespace xfindextrema {

struct Extrema {
  Extrema(const std::vector<std::size_t>& maxima_,
          const std::vector<std::size_t>& minima_,
          std::size_t zero_crossings_) {
    std::vector<std::size_t> maxima_shape_ = {maxima_.size()};
    std::vector<std::size_t> minima_shape_ = {minima_.size()};

    maxima = xt::adapt(maxima_, maxima_shape_);
    minima = xt::adapt(minima_, minima_shape_);
    zero_crossings = zero_crossings_;
  }

  xemd::array_type::tensor<std::size_t> maxima = {0};
  xemd::array_type::tensor<std::size_t> minima = {0};
  std::size_t                           zero_crossings = 0;
};

template<typename T> inline
std::size_t
FirstNonZero(const xemd::array_type::tensor<T>& x) {
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (x[i]) {
      return i;
    }
  }
  return x.size() - 1;
}

template<typename T>
xemd::xfindextrema::Extrema
FindExtrema(const xemd::array_type::tensor<T>& x) {
  std::vector<std::size_t> maxima;
  std::vector<std::size_t> minima;
  std::size_t zero_crossings = 0;

  auto dx = xemd::xutils::Diff<T>(x);

  enum GradientClassifier { RISING, FALLING };
  enum SignClassifier { POSITIVE, NEGATIVE };
  
  std::size_t i_begin = FirstNonZero(dx);
  auto gradient_cache = (dx[i_begin] > 0) ? RISING : FALLING;
  auto sign_cache = (x[i_begin] >= 0) ? POSITIVE : NEGATIVE;

  for (std::size_t i = i_begin; i < dx.size(); ++i) {
    if (!dx[i]) {
      continue;
    }

    auto this_gradient = (dx[i] > 0) ? RISING : FALLING;
    auto this_sign = (x[i + 1] >= 0) ? POSITIVE : NEGATIVE;

    if (this_gradient != gradient_cache) {
      if (this_gradient == FALLING) {
        maxima.push_back(i);
      } else {
        minima.push_back(i);
      }
    }

    if (this_sign != sign_cache) {
      zero_crossings += 1;
    }

    gradient_cache = this_gradient;
    sign_cache = this_sign;
  }

  return {maxima, minima, zero_crossings};
}

}  // namespace xfindpeaks

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
