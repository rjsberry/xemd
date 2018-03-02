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

#include "xtensor/xadapt.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xtensor.hpp"

#include "xemd/xemd.hpp"

TEST(xfindextrema, first_non_zero) {
  struct TestCase {
    std::vector<int> initializer;
    std::size_t      expected;
  };

  std::vector<TestCase> tests = {
    {{0, 1}, 1},
    {{0, 0, 1}, 2},
    {{0}, 0},
    {{1}, 0},
  };

  for (const auto& t : tests) {
    std::vector<std::size_t> shape = {t.initializer.size()};
    xt::xtensor<int, 1> x = xt::adapt(t.initializer, shape);
    ASSERT_EQ(xemd::xfindextrema::FirstNonZero<int>(x), t.expected);
  }
}

TEST(xfindextrema, check_maxima) {
  struct TestCase {
    int  a, b;
    bool expected;
  };

  std::vector<TestCase> tests = {
    {1, -1, true},
    {-1, 1, false},
    {1, 1, false},
    {-1, -1, false},
    {1, 0, false},
    {0, 1, false},
  };

  for (const auto& t : tests) {
    ASSERT_EQ(xemd::xfindextrema::CheckMaxima(t.a, t.b), t.expected);
  }
}

TEST(xfindextrema, check_minima) {
  struct TestCase {
    int  a, b;
    bool expected;
  };

  std::vector<TestCase> tests = {
    {1, -1, false},
    {-1, 1, true},
    {1, 1, false},
    {-1, -1, false},
    {1, 0, false},
    {0, 1, false},
  };

  for (const auto& t : tests) {
    ASSERT_EQ(xemd::xfindextrema::CheckMinima(t.a, t.b), t.expected);
  }
}

TEST(xfindextrema, check_zero_crossing ) {
  struct TestCase {
    int  a, b;
    bool expected;
  };

  std::vector<TestCase> tests = {
    {1, -1, true},
    {-1, 1, true},
    {1, 1, false},
    {-1, -1, false},
    {0, 1, false},
    {1, 0, false},
    {0, -1, false},
    {-1, 0, false},
  };

  for (const auto& t : tests) {
    ASSERT_EQ(xemd::xfindextrema::CheckZeroCrossing(t.a, t.b), t.expected);
  }   
}

TEST(xfindextrema, test_find_extrema) {
  struct ExtremaSTL {
    std::vector<std::size_t> maxima;
    std::vector<std::size_t> minima;
    std::size_t              zero_crossings;
  };

  struct TestCase {
    std::vector<int> initializer;
    ExtremaSTL       expected;
  };

  std::vector<TestCase> tests = {
    {{0, 1, 2, 1, 0}, {{2}, {}, 0}},
    {{2, 1, 0, 1, 2}, {{}, {2}, 0}},
    {{1, 2, -2, -1}, {{1}, {2}, 1}},
    {{1, -1, 1, -1}, {{2}, {1}, 3}},
    {{0, 1, 1, 1, 2}, {{}, {}, 0}},
    {{2, 1, 1, 1, 0}, {{}, {}, 0}},
  };

  for (const auto& t : tests) {
    std::vector<std::size_t> shape = {t.initializer.size()};
    xt::xtensor<int, 1> x = xt::adapt(t.initializer, shape);

    auto e = xemd::xfindextrema::FindExtrema(x);

    ASSERT_EQ(e.maxima.size(), t.expected.maxima.size());
    ASSERT_EQ(e.minima.size(), t.expected.minima.size());
    EXPECT_EQ(e.zero_crossings, t.expected.zero_crossings);

    for (std::size_t i = 0; i < e.maxima.size(); ++i) {
      ASSERT_EQ(e.maxima[i], t.expected.maxima[i]);
    }

    for (std::size_t i = 0; i < e.minima.size(); ++i) {
      ASSERT_EQ(e.minima[i], t.expected.minima[i]);
    }
  }
}
