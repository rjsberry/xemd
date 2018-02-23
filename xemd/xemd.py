# xemd, Copyright (c) 2018, Richard Berry <rjsberry@protonmail.com>
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.

__all__ = ['emd', 'eemd', 'ceemdan']

from functools import wraps

import xemd_core


def _validate_input_signal(function):
    """Ensures the input signal to an xemd family function meets the
    required boundary conditions.
    """
    @wraps(function)
    def input_function_wrapper(*args, **kwargs):
        if len(args) and args[0].ndim != 1:
            raise ValueError("Input signal must have shape (N,)")
        function(*args, **kwargs)
    return input_function_wrapper


def _validate_s_number(function):
    """Ensures the stopping criterion specified for an xemd family
    function meets the required specification.
    """
    @wraps(function)
    def input_function_wrapper(*args, **kwargs):
        if (kwargs.get('s_number') is not None and
            kwargs.get('s_number') < 0):
            raise ValueError("S number must be a positive integer")
        function(*args, **kwargs)
    return input_function_wrapper


def _validate_auxillary_parameters(function):
    """Ensures any additional function parameters to an xemd family
    function meet the required specification.
    """
    @wraps(function)
    def input_function_wrapper(*args, **kwargs):
        if (kwargs.get('ensemble_size') is not None and
            kwargs.get('ensemble_size') < 0):
            raise ValueError("Ensemble size must be a positive integer")
        if (kwargs.get('noise_strength') is not None and
           kwargs.get('noise_strength') < 0):
            raise ValueError("Ensemble size ")
        function(*args, **kwargs)
    return input_function_wrapper


@_validate_input_signal
@_validate_s_number
def emd(signal,
        s_number=None):
    """ Main C++ empirical mode decomposition entry point.
    """
    return xemd_core.emd(signal)


@_validate_input_signal
@_validate_s_number
@_validate_auxillary_parameters
def eemd(signal,
         s_number=None,
         ensemble_size=None,
         noise_strength=None):
    """ Main C++ ensemble empirical mode decomposition entry point.
    """
    return xemd_core.eemd(signal)


@_validate_input_signal
@_validate_s_number
@_validate_auxillary_parameters
def ceemdan(signal,
            s_number=None,
            ensemble_size=None,
            noise_strength=None):
    """ Main C++ complete ensemble empirical mode decomposition with
    adaptive noise entry point.
    """
    return xemd_core.ceemdan(signal)
