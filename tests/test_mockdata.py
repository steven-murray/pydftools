#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mockdata` module."""

import pytest

from pydftools import mockdata


def test_number():
    data = mockdata(n=100)[0]
    assert data.n_data == 100
