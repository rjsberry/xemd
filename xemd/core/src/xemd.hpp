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

#ifndef INCLUDE_XEMD_EMD_HPP_
#define INCLUDE_XEMD_EMD_HPP_

#include <xtensor-python/pyarray.hpp>

namespace xemd {

void
emd(const xt::pyarray<double>& s);

}  // namespace xemd

#endif  // INCLUDE_XEMD_EMD_HPP_
