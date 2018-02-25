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

#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>

#include <xemd/xemd.hpp>
#include "test.hpp"

TEST_CASE( "test_num_imfs", "[test_xutil]" ) {
  struct TestCaseTable {
    std::size_t input_tensor_size;
    std::size_t expected_imfs;
  };

  TestCaseTable tests[] = {
    {1, 1},
    {2, 1},
    {3, 1},
    {4, 2},
    {7, 2},
    {8, 3},
  };

  for (const auto& test : tests) {
    auto mock_array = xt::zeros<double>({test.input_tensor_size});
    REQUIRE( xemd::xutils::NumImfs<double>(mock_array) == test.expected_imfs );
  }
}

TEST_CASE ( "test_diff", "[test_xutil]" ) {
  auto linear = xt::arange<int>({10});
  auto linear_d = xemd::xutils::Diff<int>(linear);
  
  REQUIRE( linear_d.size() == linear.size() - 1 );
  for (std::size_t i = 0; i < linear.size() - 1; ++i) {
      REQUIRE( linear_d[i] == 1 );
      REQUIRE( (linear[i + 1] - linear[i]) == 1 );
  }

  auto nonlinear = xt::arange<double>({10});
  for (auto e : nonlinear) {
    e *= xt::random::rand<double>({1})[0];
  }
  auto nonlinear_d = xemd::xutils::Diff<double>(linear);

  REQUIRE( nonlinear_d.size() == nonlinear.size() - 1 );
  for (std::size_t i = 0; i < linear.size() - 1; ++i) {
      REQUIRE( nonlinear_d[i] == (nonlinear[i + 1] - nonlinear[i]) );
  }
}