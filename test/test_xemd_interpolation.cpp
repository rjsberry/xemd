// xemd: https://github.com/rjsberry/xemd
//
// Copyright (C) 2018, Richard Berry <rjsberry@protonmail.com>
//
// Distributed under the terms of BSD 2-Clause "simplified" license. (See
// accompanying file LICENSE, or copy at
// https://github.com/rjsberry/xemd/blob/master/LICENSE)
//

#include <cmath>
#include <functional>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "xtensor/xrandom.hpp"
#include "xtensor/xtensor.hpp"

#include "xemd/xemd.hpp"

#define GENERIC(x) std::function<x (x)>

TEST(interpolation, interpolator_factory_linear) {
  auto x = xt::arange<double>(0, 2);
  auto y = xt::random::randn<double>({2});
  auto s = xemd::xinterpolate::CreateInterpolator<double>(x, y);

  auto expected =
    dynamic_cast<xemd::xinterpolate::Linear<double>*>(s.get());

  ASSERT_TRUE(expected != nullptr);
}

TEST(interpolation, interpolator_factory_polynomial) {
  auto x = xt::arange<double>(0, 3);
  auto y = xt::random::randn<double>({3});
  auto s = xemd::xinterpolate::CreateInterpolator<double>(x, y);

  auto expected =
    dynamic_cast<xemd::xinterpolate::Polynomial<double>*>(s.get());

  ASSERT_TRUE(expected != nullptr);
}

TEST(interpolation, interpolator_factory_cubic_spline) {
  auto x = xt::arange<double>(0, 4);
  auto y = xt::random::randn<double>({4});
  auto s = xemd::xinterpolate::CreateInterpolator<double>(x, y);

  auto expected =
    dynamic_cast<xemd::xinterpolate::Spline<double>*>(s.get());

  ASSERT_TRUE(expected != nullptr);
}

namespace {

// Easily lets us pass `xt::xtensor` to the `GENERIC` macro.
typedef xt::xtensor<double, 1> tensor;

// The steps at which interpolators are evaluated throughout tests.
const double TEST_STEP = 0.1;

// The confidence interval at which to assert interpolated values.
const double CONFIDENCE = 0.99;

// Test whether a value lies within a certain confidence level.
bool
ApproximatesTo(double _test, double _target, double confidence) {
  auto test = std::fabs(_test);
  auto target = std::fabs(_target);

  auto lower_bound = (target * confidence);
  auto upper_bound = (target * (2 - confidence));

  if (test >= lower_bound && test <= upper_bound) {
    return true;
  }
  return false;
}

// Generate a random number in the range [0, 1].
double
Random(void) {
  return xt::random::rand<double>({1})[0];
}

// Forms the basic test case table for interpolation tests.
struct TestCase {
  GENERIC(tensor) f;
  GENERIC(double) g;
};

}  // namespace

TEST(interpolation, linear_interpolation) {
  auto x = xt::arange<double>(-1, 1);

  auto m = Random();
  auto c = Random();

  auto f_constant = [](auto x){ return x; };
  auto f_rising = [m,c](auto x){ return m * x + c; };
  auto f_falling = [m,c](auto x){ return -m * x + c; };

  std::vector<TestCase> tests = {
    {f_constant, f_constant},
    {f_rising, f_rising},
    {f_falling, f_falling},
  };

  for (const auto& test : tests) {
    auto y = test.f(x);
    auto i = xemd::xinterpolate::Linear<double>(x, y);

    for (auto k = x[0]; k < x[1]; k += TEST_STEP) {
      if (k > -TEST_STEP && k < TEST_STEP) {
        k = 0;
      }
      EXPECT_TRUE(ApproximatesTo(i(k), test.g(k), CONFIDENCE));
    }
  }
}

TEST(interpolation, polynomial_interpolation) {
  auto x = xt::linspace<double>(-1, 1, 3);

  auto a = Random();
  auto b = Random();
  auto c = Random();

  auto f_positive = [a,b,c](auto x){ return a*x*x + b*x + c; };
  auto f_negative = [a,b,c](auto x){ return -a*x*x + b*x + c; };

  std::vector<TestCase> tests = {
    {f_positive, f_positive},
    {f_negative, f_negative},
  };

  for (const auto& test : tests) {
    auto y = test.f(x);
    auto i = xemd::xinterpolate::Polynomial<double>(x, y);

    for (auto k = x[0]; k < x[2]; k += TEST_STEP) {
      if (k > -TEST_STEP && k < TEST_STEP) {
        k = 0;
      }
      EXPECT_TRUE(ApproximatesTo(i(k), test.g(k), CONFIDENCE));
    }
  }
}

namespace {

// Used to sample signal `x` values for cubic spline interpolation tests.
const double BOUND_MIN = -M_PI;
const double BOUND_MAX = M_PI;
const double NUM_SAMPLES = 100;

}  // namespace

TEST(interpolation, cubic_spline_interpolation) {
  auto x = xt::linspace<double>(BOUND_MIN, BOUND_MAX, NUM_SAMPLES);

  auto f_linear = [](auto x){ return x; };
  auto f_quadratic = [](auto x){ return x * x; };
  auto f_cubic = [](auto x){ return x * x * x; };
  auto f_sine = [](auto x){ return sin(x); };
  auto f_runge = [](auto x){ return 1 / (1 + 25 * x * x); };

  std::vector<TestCase> tests = {
    {f_linear, f_linear},
    {f_quadratic, f_quadratic},
    {f_cubic, f_cubic},
    {f_sine, f_sine},
    {f_runge, f_runge},
  };

  for (const auto& test : tests) {
    auto y = test.f(x);
    auto s = xemd::xinterpolate::Spline<double>(x, y);

    for (double k = x[0]; k < x[x.size() - 1]; k += TEST_STEP) {
      EXPECT_TRUE(ApproximatesTo(s(k), test.g(k), CONFIDENCE));
    }
  }
}
