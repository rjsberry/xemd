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

#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

#include "xemd.hpp"
#include "xeemd.hpp"
#include "xceemdan.hpp"

namespace {

const char* EMD_DOCSTRING =
  "Empirical mode decomposition";

const char* EEMD_DOCSTRING =
  "Ensemble empirical mode decomposition";

const char* CEEMDAN_DOCSTRING =
  "Complete ensemble empirical mode decomposition with adaptive noise";

}  // namespace

PYBIND11_MODULE(xemd_core, m)
{
  xt::import_numpy();

  m.doc() = "The xemd C++ back-end";

  m.def("emd", xemd::emd, EMD_DOCSTRING);
  m.def("eemd", xemd::eemd, EEMD_DOCSTRING);
  m.def("ceemdan", xemd::ceemdan, CEEMDAN_DOCSTRING);
}
