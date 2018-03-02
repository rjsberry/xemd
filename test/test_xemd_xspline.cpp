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
#include <string>

#include <fmt/format.h>

#include <xtensor/xtensor.hpp>

#include <xemd/xemd.hpp>
#include "test.hpp"

#define GENERIC(x) std::function<x ( x )>

namespace {

typedef xt::xtensor<double, 1> tensor;

const double CONFIDENCE = 0.99;

const double TEST_STEP = 0.1;

const double BOUND_MIN = -M_PI;
const double BOUND_MAX = M_PI;
const double NUM_SAMPLES = 100;

// Test whether a value lies within a certain confidence level.
bool
ApproximatesTo(double _test, double _target, double confidence) {
  auto test = fabs(_test);
  auto target = fabs(_target);
  if ((test >= (target * confidence)) &&
      (test <= (target * (2 - confidence)))) {
    return true;
  }
  return false;
}

}  // namespace

TEST_CASE( "test_spline_interpolation", "[test_xspline]" ) {
  struct TestCase {
    xt::xtensor<double, 1> x;
    GENERIC(tensor)        f;
    GENERIC(double)        g;
    std::string            label;
  };

  auto x = xt::linspace<double>(BOUND_MIN, BOUND_MAX, NUM_SAMPLES);

  auto f_linear = [](auto x){ return x; };
  auto f_quadratic = [](auto x){ return x * x; };
  auto f_cubic = [](auto x){ return x * x * x; };
  auto f_sine = [](auto x){ return sin(x); };
  auto f_runge = [](auto x){ return 1 / (1 + 25 * x * x); };

  std::vector<TestCase> tests = {
    {x, f_linear, f_linear, "f(x) = x"},
    {x, f_quadratic, f_quadratic, "f(x) = x^2"},
    {x, f_cubic, f_cubic, "f(x) = x^3"},
    {x, f_sine, f_sine, "f(x) = sin(x)"},
    {x, f_runge, f_runge, "f(x) = 1 / (1 + 25x^2)"}
  };

  for (const auto& test : tests) {
    SECTION( test.label ) {
      auto y = test.f(x);
      auto s = xemd::xspline::Spline<double>(x, y);
      for (double k = x[0]; k < x[x.size() - 1]; k += TEST_STEP) {
        auto eval = s(k);
        SECTION( fmt::format("f({}) = {} :: {}", k, test.g(k), eval) ) {
          CHECK( ApproximatesTo(eval, test.g(k), CONFIDENCE) );
        }
      }
    }
  }
}
