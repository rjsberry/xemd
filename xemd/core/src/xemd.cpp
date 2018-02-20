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

#include <numeric>

#include "pybind11/pybind11.h"
#include "xtensor/xmath.hpp"
#include "xtensor/xtensor.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

double
sum_of(xt::pyarray<double>& s)
{
  auto arr = xt::xtensor<double, 1>(s);
  return std::accumulate(arr.begin(), arr.end(), 0.0);
}

PYBIND11_MODULE(xemd_core, m)
{
  xt::import_numpy();
  m.doc() = "xemd hello world with xtensor-python/pybind11";
  m.def("sum_of", sum_of, "Sum the given 1-D array");
}
