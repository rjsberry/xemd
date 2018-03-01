// xemd: https://github.com/rjsberry/xemd
//
// Copyright (C) 2018, Richard Berry <rjsberry@protonmail.com>
//
// Distributed under the terms of BSD 2-Clause "simplified" license. (See
// accompanying file LICENSE, or copy at
// https://github.com/rjsberry/xemd/blob/master/LICENSE)
//

#include <xtensor/xtensor.hpp>

#include <xemd/xemd.hpp>
#include "test.hpp"

const double FLOATING_POINT_ERROR_TOLERANCE = 0.333;

namespace {

bool
ApproximatesTo(double test, double target, double tolerance) {
  if ((test > target - tolerance) && (test < target + tolerance)) {
    return true;
  }
  return false;
}

}  // namespace

TEST_CASE( "test_spline_interpolation", "[test_xspline]" ) {
  // Quadratic.
  xt::xtensor<double, 1> X = {-10, -7, -3,  0,  1,  2,   4,  7,  8,  10,  12,  13};
  xt::xtensor<double, 1> Y = {100, 49,  9,  0,  1,  4,  16, 49, 64, 100, 144, 169};

  auto s = xemd::xspline::Spline<double>(X, Y);

  CHECK( ApproximatesTo(s( 3),   9, FLOATING_POINT_ERROR_TOLERANCE) );
  CHECK( ApproximatesTo(s( 5),  25, FLOATING_POINT_ERROR_TOLERANCE) );
  CHECK( ApproximatesTo(s( 6),  36, FLOATING_POINT_ERROR_TOLERANCE) );
  CHECK( ApproximatesTo(s( 9),  81, FLOATING_POINT_ERROR_TOLERANCE) );
  CHECK( ApproximatesTo(s(11), 121, FLOATING_POINT_ERROR_TOLERANCE) );
}
