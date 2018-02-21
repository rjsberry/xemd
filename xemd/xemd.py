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
        function(*args, **kwargs)
    return input_function_wrapper


def _validate_stopping_criterion(function):
    """Ensures the stopping criterion specified for an xemd family
    function meets the required specification.
    """
    @wraps(function)
    def input_function_wrapper(*args, **kwargs):
        function(*args, **kwargs)
    return input_function_wrapper


def _validate_auxillary_parameters(function):
    """Ensures any additional function parameters to an xemd family
    function meet the required specification.
    """
    @wraps(function)
    def input_function_wrapper(*args, **kwargs):
        function(*args, **kwargs)
    return input_function_wrapper


@_validate_input_signal
@_validate_stopping_criterion
def emd(signal,
        stopping_criterion=None):
    """ Main C++ empirical mode decomposition entry point.
    """
    return xemd_core.emd(signal)


@_validate_input_signal
@_validate_stopping_criterion
@_validate_auxillary_parameters
def eemd(signal,
         stopping_criterion=None,
         ensemble_size=None,
         noise_strength=None):
    """ Main C++ ensemble empirical mode decomposition entry point.
    """
    return xemd_core.eemd(signal)


@_validate_input_signal
@_validate_stopping_criterion
@_validate_auxillary_parameters
def ceemdan(signal,
            stopping_criterion=None,
            ensemble_size=None,
            noise_strength=None):
    """ Main C++ complete ensemble empirical mode decomposition with
    adaptive noise entry point.
    """
    return xemd_core.ceemdan(signal)
