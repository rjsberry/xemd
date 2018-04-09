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
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>

#include <unistd.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>

#include <xtensor-blas/xlinalg.hpp>

#include <xtensor-interpolate/xinterpolate.hpp>

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
void
InsertRow(xemd::array_type::array<T>& X,
          std::size_t row,
          const xemd::array_type::tensor<T>& x) {
  for (std::size_t col = 0; col < X.shape()[1]; ++col) {
    X(row, col) = x(col);
  }
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

template<typename T> inline
void
CorrectEndpoints(const xemd::array_type::tensor<T>& x,
                 xemd::array_type::tensor<T>& y,
                 const std::function<bool (T, T)>& comparator) {
  assert(x.size() == y.size());

  if (x.size() <= 2) {
    return;
  }

  auto lhs_candidate = Extrapolate(
    x[1], y[1], x[2], y[2], x[0]
  );
  auto rhs_candidate = Extrapolate(
    x[x.size()-3], y[y.size()-3], x[x.size()-2], y[y.size()-2], x[x.size()-1]
  );

  if (comparator(lhs_candidate, y[0])) {
    y[0] = lhs_candidate;
  }
  if (comparator(rhs_candidate, y[y.size()-1])) {
    y[y.size()-1] = rhs_candidate;
  }
}

}  // namespace xutils

namespace xfindextrema {

struct Extrema {
  Extrema(void) {
  }

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

template<typename T> inline
bool
CheckMaxima(T dx0, T dx) {
  if (dx0 > 0 && dx < 0) {
    return true;
  }
  return false;
}

template<typename T> inline
bool
CheckMinima(T dx0, T dx) {
  if (dx0 < 0 && dx > 0) {
    return true;
  }
  return false;
}

template<typename T> inline
bool
CheckZeroCrossing(T x0, T x) {
  if ((x0 < 0 && x > 0) || (x0 > 0 && x < 0)) {
    return true;
  }
  return false;
}

template<typename T>
xemd::xfindextrema::Extrema
FindExtrema(const xemd::array_type::tensor<T>& x) {
  std::vector<std::size_t> maxima;
  std::vector<std::size_t> minima;
  std::size_t zero_crossings = 0;

  auto dx = xemd::xutils::Diff<T>(x);

  for (std::size_t i = 0; i < dx.size() - 1; ++i) {
    if (CheckMaxima<T>(dx[i], dx[i + 1])) {
      maxima.push_back(i + 1);
    } else if (CheckMinima<T>(dx[i], dx[i + 1])) {
      minima.push_back(i + 1);
    }
  }
  for (std::size_t i = 0; i < x.size() - 1; ++i) {
    if (CheckZeroCrossing<T>(x[i], x[i + 1])) {
      ++zero_crossings;
    }
  }

  return {maxima, minima, zero_crossings};
}

}  // namespace xfindpeaks

template<typename T>
class IMF {
public:
  IMF(const xemd::array_type::tensor<T>& x)
    : xsignal(x), xsignal_x(xt::arange<double>(x.size())) {
  }

  void
  Decompose(unsigned int s_number, std::size_t maximum_iterations) {
    unsigned int s_count = 0;
    auto extrema = xfindextrema::Extrema();

    for (std::size_t i = 0; i < maximum_iterations; ++i) {
      auto previous_extrema = extrema;
      extrema = xfindextrema::FindExtrema(xsignal);

      if (StopSifting(s_number, &s_count, extrema, previous_extrema)) {
        break;
      }

      // Interpolate envelope over signal maxima.

      xemd::array_type::tensor<std::size_t> maxima_x = xt::concatenate(xtuple(
        xt::zeros<double>({ 1 }),
        extrema.maxima,
        xt::view(xsignal, xt::range(xsignal.size() - 2, xsignal.size() - 1))
      ));
      xemd::array_type::tensor<T> maxima_y = xt::index_view(xsignal, maxima_x);

      xutils::CorrectEndpoints<T>(maxima_x, maxima_y, [](T x, T y){ return x > y; });

      auto maxima_k = 3;
      if (maxima_x.size() == 2) {
        maxima_k = 1;
      } else if (maxima_x.size() == 3) {
        maxima_k = 2;
      }

      auto maxima_tck = xt::interpolate::splrep(
        static_cast<xemd::array_type::tensor<double>>(maxima_x),
        maxima_y,
        maxima_k
      );

      xt::xtensor<double, 1> maxima_spline =
        xt::interpolate::splev(xsignal_x, maxima_tck);

      // Interpolate envelope over signal minima.

      xemd::array_type::tensor<std::size_t> minima_x = xt::concatenate(xtuple(
        xt::zeros<double>({ 1 }),
        extrema.minima,
        xt::view(xsignal, xt::range(xsignal.size() - 2, xsignal.size() - 1))
      ));
      xemd::array_type::tensor<T> minima_y = xt::index_view(xsignal, minima_x);

      xutils::CorrectEndpoints<T>(minima_x, minima_y, [](T x, T y){ return x < y; });

      auto minima_k = 1;
      if (minima_x.size() == 2) {
        minima_k = 1;
      } else if (minima_x.size() == 3) {
        minima_k = 2;
      }

      auto minima_tck = xt::interpolate::splrep(
        static_cast<xemd::array_type::tensor<double>>(minima_x),
        minima_y,
        minima_k
      );

      xt::xtensor<double, 1> minima_spline =
        xt::interpolate::splev(xsignal_x, minima_tck);

      xsignal -= 0.5 * (maxima_spline + minima_spline);
    }
  }

  xemd::array_type::tensor<T>
  Extract(void) {
    return xsignal;
  }

  bool
  IsMonotonic(void) {
    if (xt::all(xutils::Diff<T>(xsignal) >= 0) ||
        xt::all(xutils::Diff<T>(xsignal) <= 0)) {
      return true;
    }
    return false;
  }

private:
  xemd::array_type::tensor<T>       xsignal;
  const xemd::array_type::tensor<T> xsignal_x;

  bool
  StopSifting(unsigned int s_number,
              unsigned int *s_count,
              const xfindextrema::Extrema& this_extrema,
              const xfindextrema::Extrema& prev_extrema) {
    auto delta_max =
      std::abs(this_extrema.maxima.size() - prev_extrema.maxima.size());
    auto delta_min =
      std::abs(this_extrema.minima.size() - prev_extrema.minima.size());
    auto delta_zc =
      std::abs(this_extrema.zero_crossings - prev_extrema.zero_crossings);

    if (delta_max + delta_min + delta_zc <= 1) {
      if (++(*s_count) >= s_number) {
        return true;
      }
    } else {
      *s_count = 0;
    }

    return false;
  }
};

template<typename T>
void
emd(const xemd::array_type::tensor<T>& xin,
    unsigned int s_number,
    std::size_t max_siftings,
    xemd::array_type::array<T>& imfs) {
  auto N = xin.size();
  auto M = xutils::NumImfs<T>(xin);

  assert(imfs.shape()[0] >= M);
  assert(imfs.shape()[1] == N);

  auto residual = xin;
  for (std::size_t i = 0; i < M; ++i) {
    auto imf = IMF<T>(residual);

    if (imf.IsMonotonic()) {
      xutils::InsertRow(imfs, i, imf.Extract());
      break;
    }

    imf.Decompose(s_number, max_siftings);
    xutils::InsertRow(imfs, i, imf.Extract());
    residual -= imf.Extract();
  }
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
