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

from datetime import datetime

import numpy as np
import pytest

import xemd

RNG_SEED = datetime.now().second


@pytest.fixture
def assert_called():
    class AssertCall(object):
        def __init__(self):
            self.called = False
        def __call__(self, *args, **kwargs):
            self.called = True
    return AssertCall()


def test_emd_np_array_one_dim_assertions(monkeypatch, assert_called):
    monkeypatch.setattr("xemd_core.emd", assert_called)
    xemd.emd(np.random.rand(np.random.randint(RNG_SEED + 1)))
    assert assert_called.called


def test_eemd_np_array_one_dim_assertions(monkeypatch, assert_called):
    monkeypatch.setattr("xemd_core.eemd", assert_called)
    xemd.eemd(np.random.rand(np.random.randint(RNG_SEED + 1)))
    assert assert_called.called


def test_ceemdan_np_array_one_dim_assertions(monkeypatch, assert_called):
    monkeypatch.setattr("xemd_core.ceemdan", assert_called)
    xemd.ceemdan(np.random.rand(np.random.randint(RNG_SEED + 1)))
    assert assert_called.called


def test_emd_np_array_multi_dim_assertions(monkeypatch, assert_called):
    monkeypatch.setattr("xemd_core.emd", assert_called)
    with pytest.raises(ValueError):
        xemd.eemd(np.random.rand(np.random.randint(RNG_SEED + 1), 2))
    assert not assert_called.called


def test_eemd_np_array_multi_dim_assertions(monkeypatch, assert_called):
    monkeypatch.setattr("xemd_core.eemd", assert_called)
    with pytest.raises(ValueError):
       xemd.eemd(np.random.rand(np.random.randint(RNG_SEED + 1), 2))
    assert not assert_called.called


def test_ceemdan_np_array_multi_dim_assertions(monkeypatch, assert_called):
    monkeypatch.setattr("xemd_core.ceemdan", assert_called)
    with pytest.raises(ValueError):
        xemd.ceemdan(np.random.rand(np.random.randint(RNG_SEED + 1), 2))
    assert not assert_called.called
