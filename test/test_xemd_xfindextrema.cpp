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

#include <vector>

#include <xtensor/xadapt.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>

#include <xemd/xemd.hpp>
#include "test.hpp"

TEST_CASE( "test_first_non_zero", "[test_xfindextrema]" ) {
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

    REQUIRE( xemd::xfindextrema::FirstNonZero<int>(x) == t.expected );
  }
}

TEST_CASE( "test_find_extrema", "[test_xfindextrema]" ) {
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

    REQUIRE( e.maxima.size() == t.expected.maxima.size() );
    for (std::size_t i = 0; i < e.maxima.size(); ++i) {
      REQUIRE( e.maxima[i] == t.expected.maxima[i] );
    }

    REQUIRE( e.minima.size() == t.expected.minima.size() );
    for (std::size_t i = 0; i < e.minima.size(); ++i) {
      REQUIRE( e.minima[i] == t.expected.minima[i] );
    }

    REQUIRE( e.zero_crossings == t.expected.zero_crossings );
  }
}
