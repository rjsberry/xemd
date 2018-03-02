// xemd: https://github.com/rjsberry/xemd
//
// Copyright (C) 2018, Richard Berry <rjsberry@protonmail.com>
//
// Distributed under the terms of BSD 2-Clause "simplified" license. (See
// accompanying file LICENSE, or copy at
// https://github.com/rjsberry/xemd/blob/master/LICENSE)
//

#include "gtest/gtest.h"

#include "xtensor/xrandom.hpp"
#include "xtensor/xtensor.hpp"

#include "xemd/xemd.hpp"

const std::size_t ARRAY_LENGTH = 10;

TEST(xutils, diff) {
  auto linear = xt::arange<int>({ARRAY_LENGTH});
  auto linear_d = xemd::xutils::Diff<int>(linear);
  
  ASSERT_EQ(linear_d.size(), linear.size() - 1);
  for (std::size_t i = 0; i < linear.size() - 1; ++i) {
    ASSERT_EQ(linear_d[i], 1);
    ASSERT_EQ((linear[i + 1] - linear[i]), 1);
  }

  auto nonlinear = xt::arange<double>({ARRAY_LENGTH});
  for (auto e : nonlinear) {
    e *= xt::random::rand<double>({1})[0];
  }
  auto nonlinear_d = xemd::xutils::Diff<double>(linear);

  ASSERT_EQ(nonlinear_d.size(), nonlinear.size() - 1);
  for (std::size_t i = 0; i < linear.size() - 1; ++i) {
    ASSERT_EQ(nonlinear_d[i], (nonlinear[i + 1] - nonlinear[i]));
  }
}

TEST(xutils, linear_extrapolation) {
  struct TestCase {
    int x0, y0, x1, y1, x2, y2;
  };

  std::vector<TestCase> tests = {
    {0, 0, 1, 0, 2, 0},
    {1, 0, 2, 0, 0, 0},
    {2, 0, 1, 0, 0, 0},
    {0, 0, 1, 1, 2, 2},
    {1, 1, 0, 0, 2, 2},
    {2, 2, 1, 1, 0, 0},
    {0, 2, 1, 1, 2, 0},
    {1, 1, 0, 2, 2, 0},
    {2, 0, 1, 1, 0, 2},
  };

  for (const auto& t : tests) {
    ASSERT_EQ(xemd::xutils::Extrapolate<int>(t.x0, t.y0, t.x1, t.y1, t.x2), t.y2);
  }
}

TEST(xutils, monotonic) {
  auto constant = xt::ones<int>({ARRAY_LENGTH});
  ASSERT_TRUE(xemd::xutils::IsMonotonic<int>(constant));

  auto linear = xt::arange<int>(-static_cast<int>(ARRAY_LENGTH)/2,
                                static_cast<int>(ARRAY_LENGTH)/2);
  ASSERT_TRUE(xemd::xutils::IsMonotonic<int>(linear));

  auto noise = xt::random::randn<double>({ARRAY_LENGTH});
  ASSERT_FALSE(xemd::xutils::IsMonotonic<double>(noise));

  auto quadratic = linear * linear;
  ASSERT_FALSE(xemd::xutils::IsMonotonic<int>(quadratic));

  auto cubic = linear * linear * linear;
  ASSERT_TRUE(xemd::xutils::IsMonotonic<int>(cubic));
}

TEST(xutils, num_imfs) {
  struct TestCase {
    std::size_t input_tensor_size;
    std::size_t expected_imfs;
  };

  std::vector<TestCase> tests = {
    {1, 1},
    {2, 1},
    {3, 1},
    {4, 2},
    {7, 2},
    {8, 3},
  };

  for (const auto& test : tests) {
    auto mock_array = xt::zeros<double>({test.input_tensor_size});
    ASSERT_EQ(xemd::xutils::NumImfs<double>(mock_array), test.expected_imfs);
  }
}
