"""Shared fixtures, and a session gate on the compiled library.

Every test in this suite asserts on the output of a kernel in
`dist/libmojo-pyroomacoustics.so`. If that library is missing the session gate
fails loudly instead of letting a pure-Python path satisfy the assertions.
"""

import numpy as np
import pytest

from mojopyroomacoustics import _lib

pra = pytest.importorskip("pyroomacoustics")


@pytest.fixture(scope="session", autouse=True)
def compiled_library():
    """Fail the whole session if the compiled Mojo library cannot be loaded."""
    return _lib.require()


@pytest.fixture(scope="session")
def lib(compiled_library):
    return compiled_library


@pytest.fixture(scope="session")
def pra_libroom():
    return pytest.importorskip("pyroomacoustics.libroom")


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def rel_err(ours, theirs):
    """Max absolute difference relative to the magnitude of the reference."""
    ours = np.asarray(ours)
    theirs = np.asarray(theirs)
    scale = np.max(np.abs(theirs))
    if scale == 0:
        return float(np.max(np.abs(ours)))
    return float(np.max(np.abs(ours - theirs)) / scale)


@pytest.fixture
def error():
    return rel_err


def f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def c128(a):
    return np.ascontiguousarray(a, dtype=np.complex128)


# `_lib.addr`, not `a.ctypes.data`: the address must keep the buffer alive. A
# bare int would let `addr(f64([0.5]))` free the temporary the instant `addr`
# returned, and the kernel would then read freed memory. Several tests below
# deliberately pass temporaries that way, so they exercise this.
addr = _lib.addr