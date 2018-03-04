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
#include <memory>
#include <type_traits>
#include <vector>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include <xtensor-blas/xlinalg.hpp>

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

namespace xinterpolate {

template<typename T>
class Interpolator {
public:
  virtual T operator() (T x) const = 0;
};

template<typename T>
class Linear : public Interpolator<T> {
public:
  Linear(const xemd::array_type::tensor<T>& x,
         const xemd::array_type::tensor<T>& y)
    : x0(x[0])
    , y0(y[0])
    , x1(x[1])
    , y1(y[1]) {
  }

  T operator() (T x) const {
    return xutils::Extrapolate(x0, y0, x1, y1, x);
  }

private:
  T x0, y0, x1, y1;
};

template<typename T>
class Polynomial : public Interpolator<T> {
private:
  struct Coefficients {
    T a, b, c;
  };

public:
  Polynomial(const xemd::array_type::tensor<T>& x,
             const xemd::array_type::tensor<T>& y) 
    : coeffs(CalculateCoefficients(x, y)) {
  }

  T operator() (T x) const {
    return coeffs.a*x*x + coeffs.b*x + coeffs.c;
  }

private:
  const Coefficients coeffs;

  static
  Coefficients CalculateCoefficients(const xemd::array_type::tensor<T>& x,
                                     const xemd::array_type::tensor<T>& y) {
    // FIXME: Update to `xt::xtensorf<T, xt::shape<3, 3>>`
    xemd::array_type::array<T> vandermonde
      {{x[0] * x[0], x[0], 1},
       {x[1] * x[1], x[1], 1},
       {x[2] * x[2], x[2], 1}};

    auto raw_coeffs = xt::linalg::solve(vandermonde, y);

    return {raw_coeffs[0], raw_coeffs[1], raw_coeffs[2]};
  }
};

template<typename T>
class Spline : public Interpolator<T> {
private:
  struct SplineCoefficients {
    xemd::array_type::tensor<T> a, b, c, d, x;
  };

public:
  Spline(const xemd::array_type::tensor<T>& x,
         const xemd::array_type::tensor<T>& y) 
      : coefficients_(CalculateSplineCoefficients(x, y)) {
  }

  T operator()(T x) const {
    return EvaluateCoefficients_(x);
  }

private:
  const SplineCoefficients coefficients_;

  static SplineCoefficients
  CalculateSplineCoefficients(const xemd::array_type::tensor<T>& x,
                              const xemd::array_type::tensor<T>& y) {
    auto N = x.size();
    auto h = xutils::Diff<T>(x);

    xemd::array_type::tensor<T> a = xt::zeros<T>({N - 2});
    xemd::array_type::tensor<T> b = xt::zeros<T>({N});
    xemd::array_type::tensor<T> g = xt::zeros<T>({N});
    for (std::size_t i = 0; i < N - 1; ++i) {
      b[i] = (y[i+1] - y[i]) / h[i];
    }
    for (std::size_t i = 0; i < N - 2; ++i) {
      a[i] = 2 * (h[i+1] + h[i]);  // System matrix diagonal.
      g[i+1] = b[i+1] - b[i];
    }

    // Construct the tri-diagonal matrix for the "periodic" boundary condition.
    auto diag =
      // Diagonal
      xt::diag(a) +
      // Off-diagonal (upper)
      xt::diag(xt::view(h, xt::range(0, h.size() - 2)), 1) +
      // Off-diagonal (lower)
      xt::diag(xt::view(h, xt::range(0, h.size() - 2)), -1) +
      // Corner (upper right)
      xt::diag(xt::view(h, xt::range(h.size() - 2, h.size() - 1)), a.size() - 1) +
      // Corner (lower left)
      xt::diag(xt::view(h, xt::range(h.size() - 2, h.size() - 1)), -(a.size() - 1))
    ;
    // Solve the constructed tridiagonal system matrix with `g` for `c`.
    xemd::array_type::tensor<T> c = xt::concatenate(xtuple(
      xt::zeros<T>({1}),
      xt::linalg::solve(diag, xt::view(g, xt::range(1, g.size() - 1))),
      xt::zeros<T>({1})
    ));

    xemd::array_type::tensor<T> d = xt::zeros<T>({N});
    for (std::size_t i = 0; i < N - 1; ++i) {
      if (i < N - 2) {
        d[i] = (c[i+1] - c[i]) / h[i];
        b[i] -= h[i] * (c[i+1] + 2 * c[i]);
      } else {
        d[i] = -(c[i] / h[i]);
        b[i] -= 2 * c[i] * h[i];
      }
      c[i] *= 3.;
    }

    return {y, b, c, d, x};
  }

  T EvaluateCoefficients_(T x) const {
    T h;
    std::size_t i = coefficients_.x.size() - 2;
    for (; i < coefficients_.x.size() - 1; --i) {
      h = x - coefficients_.x[i];
      if (h >= 0) {
        break;
      }
    }
    auto y = coefficients_.d[i];
    y = y * h + coefficients_.c[i];
    y = y * h + coefficients_.b[i];
    y = y * h + coefficients_.a[i];

    return y;
  }
};

template<typename T>
std::unique_ptr<Interpolator<T>> CreateInterpolator(
    const xemd::array_type::tensor<T>& x,
    const xemd::array_type::tensor<T>& y) {
  assert(x.size() == y.size());
  assert(x.size() > 1);

  std::unique_ptr<Interpolator<T>> ptr;
  switch (x.size()) {
  case 2:
    ptr = std::make_unique<Linear<T>>(x, y);
    return ptr;
  case 3:
    ptr = std::make_unique<Polynomial<T>>(x, y);
    return ptr;
  default:
    ptr = std::make_unique<Spline<T>>(x, y);
    return ptr;
  }
}

}  // namespace xinterpolate

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
