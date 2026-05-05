# Copyright 2025 UChicago Argonne, LLC.
# (c.f. AUTHORS, LICENSE)
#
# This file is part of the dstm project.
# For more info, see https://github.com/anlsys/dstm
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import platform
import ctypes as ct

import numpy as np
import numpy.typing as npt

if os.environ.get("LIBDSTM_SO_"):
    libnrm = ct.cdll.LoadLibrary(os.environ.get("LIBDSTM_SO_"))  # type: ignore
else:
    if platform.uname()[0] == "Linux":
        libnrm = ct.cdll.LoadLibrary("libdstm.so")
    else:
        libnrm = ct.cdll.LoadLibrary("libdstm.dylib")


# struct dstm_darray_s {
# 	size_t element_size;
# 	size_t total_elements;
# 	size_t local_elements;
# 	dstm_region_t *region;
# 	/* per-rank offset from the global indices */
# 	size_t *rank_offsets;
# 	/* per-rank size of the local part, needed for gather operations */
# 	size_t *rank_elements;
# 	/* per-rank ptr for the local part */
# 	dstm_ptr_t *rank_ptrs;
# };

class _Darray(ct.Structure):
    _fields_ = [
        # byte size of each element
        ("element_size", ct.c_size_t),
        # total number of elements
        ("n_total_elements", ct.c_size_t),
        # number local elements
        ("n_local_elements", ct.c_size_t),
        # shouldn't need for python
        ("__region", ct.c_void_p),

        # offsets by rank
        ("rank_offsets", ct.POINTER(ct.c_size_t)),
        # number of elements by rank
        ("rank_n_elements", ct.POINTER(ct.c_size_t)),
        ("__rank_ptrs", ct.c_void_p)
    ]


class Darray:

    def __init__(self, handle: ct._Pointer, dtype: npt.DTypeLike):
        self.darray = handle.contents
        self.handle = handle
        self.dtype = dtype

    @property
    def element_size(self):
        return self.darray.element_size
    
    @property
    def n_total_elements(self):
        return self.darray.n_total_elements
    
    @property
    def n_local_elements(self):
        return self.darray.n_local_elements
    
    @property
    def rank_offsets(self):
        return self.darray.rank_offsets
    
    @property
    def rank_n_elements(self):
        return self.darray.rank_n_elements


libnrm.dstm_darray_create.restype = ct.c_int
libnrm.dstm_darray_create.argtypes = [
    # ct.POINTER(ct.c_void_p),
    ct.POINTER(ct.POINTER(_Darray)),
    ct.c_size_t,
    ct.c_size_t,
]

libnrm.dstm_darray_put.restype = ct.c_int
libnrm.dstm_darray_put.argtypes = [
    ct.c_void_p,
    ct.c_size_t,
    ct.c_void_p,
    ct.c_int64,
    ct.c_int64,
]

libnrm.dstm_darray_get.restype = ct.c_int
libnrm.dstm_darray_get.argtypes = [
    ct.c_void_p,
    ct.c_size_t,
    ct.c_void_p,
    ct.c_int64,
    ct.c_int64,
]

libnrm.dstm_darray_allgatherv.restype = ct.c_int
libnrm.dstm_darray_allgatherv.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int64,
    ct.POINTER(ct.c_int64),
]

libnrm.dstm_init.restype = ct.c_int
libnrm.dstm_init.argtypes = [
    ct.POINTER(ct.c_int),
    ct.POINTER(ct.POINTER(ct.c_char_p))
    # ct.c_void_p,
    # ct.c_void_p
]

libnrm.dstm_finalize.restype = ct.c_int
libnrm.dstm_finalize.argtypes = []

libnrm.dstm_fini.restype = ct.c_int
libnrm.dstm_fini.argtypes = []

libnrm.dstm_log_setlevel.restype = ct.c_int
libnrm.dstm_log_setlevel.argtypes = [ct.c_int]

libnrm.dstm_distance_create_ring.restype = ct.c_int
libnrm.dstm_distance_create_ring.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]

libnrm.dstm_distance_get.restype = ct.c_int
libnrm.dstm_distance_get.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]

libnrm.dstm_distance_create_complete.restype = ct.c_int
libnrm.dstm_distance_create_complete.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]

libnrm.dstm_distance_destroy.restype = None
libnrm.dstm_distance_destroy.argtypes = [ct.POINTER(ct.c_void_p)]

libnrm.dstm_start.restype = ct.c_int
libnrm.dstm_start.argtypes = [ct.c_void_p]


LOG_QUIET = 0
LOG_ERROR = 1
LOG_WARNING = 2
LOG_NORMAL = 3
LOG_INFO = 4
LOG_DEBUG = 5


class Distance:
    def __init__(self, handle):
        self.handle = handle


def _check(rc, name):
    if rc != 0:
        raise RuntimeError(f"{name} failed with code {rc}")


def darray_create(dtype, local_elements):
    dtype = np.dtype(dtype)
    handle = ct.POINTER(_Darray)()
    rc = libnrm.dstm_darray_create(ct.byref(handle), dtype.itemsize, local_elements)
    _check(rc, "dstm_darray_create")
    darray = Darray(handle, dtype)
    return darray


def darray_put(darray, global_index, value, time, delta):
    arr = np.ascontiguousarray(value, dtype=darray.dtype)
    rc = libnrm.dstm_darray_put(
        darray.handle, global_index, arr.ctypes.data, time, delta
    )
    _check(rc, "dstm_darray_put")


def darray_get(darray, global_index, dst, time, delta):
    rc = libnrm.dstm_darray_get(
        darray.handle, global_index, dst.ctypes.data, time, delta
    )
    _check(rc, "dstm_darray_get")


def darray_allgatherv(darray, src, dst, time, deltas):
    src = np.ascontiguousarray(src, dtype=darray.dtype)
    if not (isinstance(dst, np.ndarray) and dst.dtype == darray.dtype and dst.flags.c_contiguous):
        raise TypeError("dst must be a contiguous numpy array with darray.dtype")
    deltas = np.ascontiguousarray(deltas, dtype=np.int64)
    rc = libnrm.dstm_darray_allgatherv(
        darray.handle,
        src.ctypes.data,
        dst.ctypes.data,
        time,
        deltas.ctypes.data_as(ct.POINTER(ct.c_int64)),
    )
    _check(rc, "dstm_darray_allgatherv")


def init(argv=None):
    # if argv is None:
    #     argv = sys.argv
    # encoded = [s.encode() for s in argv]
    # argv_arr = (ct.c_char_p * (len(encoded) + 1))(*encoded, None)
    # argv_pp = ct.cast(argv_arr, ct.POINTER(ct.c_char_p))
    # argc = ct.c_int(len(encoded))
    # rc = libnrm.dstm_init(ct.byref(argc), ct.byref(argv_pp))
    rc = libnrm.dstm_init(None, None)
    _check(rc, "dstm_init")


def finalize():
    rc = libnrm.dstm_finalize()
    _check(rc, "dstm_finalize")


def fini():
    rc = libnrm.dstm_fini()
    _check(rc, "dstm_fini")


def log_setlevel(level):
    rc = libnrm.dstm_log_setlevel(level)
    _check(rc, "dstm_log_setlevel")


def distance_get(dist: Distance, i: int, j: int) -> int:
    distance = libnrm.dstm_distance_get(dist.handle, i, j)
    return distance


def distance_create_ring(size):
    handle = ct.c_void_p()
    rc = libnrm.dstm_distance_create_ring(ct.byref(handle), size)
    _check(rc, "dstm_distance_create_ring")
    return Distance(handle)


def distance_create_complete(size):
    handle = ct.c_void_p()
    rc = libnrm.dstm_distance_create_complete(ct.byref(handle), size)
    _check(rc, "dstm_distance_create_complete")
    return Distance(handle)


def distance_destroy(dist):
    libnrm.dstm_distance_destroy(ct.byref(dist.handle))


def start(dist):
    rc = libnrm.dstm_start(dist.handle)
    _check(rc, "dstm_start")
