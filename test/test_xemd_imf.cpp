// xemd: https://github.com/rjsberry/xemd
//
// Copyright (C) 2018, Richard Berry <rjsberry@protonmail.com>
//
// Distributed under the terms of BSD 2-Clause "simplified" license. (See
// accompanying file LICENSE, or copy at
// https://github.com/rjsberry/xemd/blob/master/LICENSE)
//

#include <vector>

#include "gtest/gtest.h"
#include "xtensor/xrandom.hpp"
#include "xtensor/xtensor.hpp"

#include "xemd/xemd.hpp"

namespace {

const std::size_t TENSOR_LENGTH = 100;

}  // namespace

TEST(imf, extract_internal_signal) {
  xt::xtensor<double, 1> x = xt::random::randn<double>({TENSOR_LENGTH});
  auto imf = xemd::IMF<double>(&x);

  ASSERT_EQ(imf.Extract(), x);
}

TEST(imf, decompose) {
  xt::xtensor<double, 1> x = xt::random::randn<double>({TENSOR_LENGTH});
  auto x_copy = x;
  auto imf = xemd::IMF<double>(&x);
  imf.Decompose(1, 1000);

  ASSERT_NE(imf.Extract(), x_copy);
  ASSERT_EQ(imf.Extract(), x); 
}

TEST(imf, monotonic_signal_check) {
  struct TestCase {
    xt::xtensor<double, 1> x;
    bool                   is_monotonic;
  };

  xt::xtensor<double, 1> x_constant =
    xt::ones<double>({TENSOR_LENGTH});

  xt::xtensor<double, 1> x_linear =
    xt::arange<double>(-static_cast<int>(TENSOR_LENGTH)/2,
                       static_cast<int>(TENSOR_LENGTH)/2);

  xt::xtensor<double, 1> x_noise =
    xt::random::randn<double>({TENSOR_LENGTH});

  xt::xtensor<double, 1> x_quadratic =
    x_linear * x_linear;

  xt::xtensor<double, 1> x_cubic =
    x_linear * x_linear * x_linear;

  std::vector<TestCase> tests = {
    {x_constant, true},
    {x_linear, true},
    {x_noise, false},
    {x_quadratic, false},
    {x_cubic, true},
  };

  for (auto& test : tests) {
    auto imf = xemd::IMF<double>(&test.x);
    ASSERT_EQ(imf.IsMonotonic(), test.is_monotonic);
  }
}
