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

#include <iostream>

#include <xtensor-python/pytensor.hpp>

namespace xemd {

void
ceemdan(const xt::pytensor<double, 1>& s)
{
  std::cout << "CORE: `xemd::ceemdan` called" << std::endl;
}

}  // namespace xemd
