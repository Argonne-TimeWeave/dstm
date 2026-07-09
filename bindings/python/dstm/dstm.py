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
    libdstm = ct.cdll.LoadLibrary(os.environ.get("LIBDSTM_SO_"))  # type: ignore
else:
    if platform.uname()[0] == "Linux":
        libdstm = ct.cdll.LoadLibrary("libdstm.so")
    else:
        libdstm = ct.cdll.LoadLibrary("libdstm.dylib")


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


libdstm.dstm_darray_create.restype = ct.c_int
libdstm.dstm_darray_create.argtypes = [
    # ct.POINTER(ct.c_void_p),
    ct.POINTER(ct.POINTER(_Darray)),
    ct.c_size_t,
    ct.c_size_t,
]

libdstm.dstm_darray_put.restype = ct.c_int
libdstm.dstm_darray_put.argtypes = [
    ct.c_void_p,
    ct.c_size_t,
    ct.c_void_p,
    ct.c_int64,
    ct.c_int64,
]

libdstm.dstm_darray_get.restype = ct.c_int
libdstm.dstm_darray_get.argtypes = [
    ct.c_void_p,
    ct.c_size_t,
    ct.c_void_p,
    ct.c_int64,
    ct.c_int64,
]

libdstm.dstm_darray_allgatherv.restype = ct.c_int
libdstm.dstm_darray_allgatherv.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int64,
    ct.POINTER(ct.c_int64),
]

libdstm.dstm_darray_get_range.restype = ct.c_int
libdstm.dstm_darray_get_range.argtypes = [
    ct.c_void_p,                # dstm_darray_t *darray
    ct.POINTER(ct.c_size_t),    # const size_t *indices
    ct.c_size_t,                # size_t count
    ct.c_void_p,                # void *dst
    ct.c_int64,                 # dstm_time_t time
    ct.c_int64,                 # dstm_time_t delta
]

libdstm.dstm_darray_put_range.restype = ct.c_int
libdstm.dstm_darray_put_range.argtypes = [
    ct.c_void_p,                # dstm_darray_t *darray
    ct.POINTER(ct.c_size_t),    # const size_t *indices
    ct.c_size_t,                # size_t count
    ct.c_void_p,                # const void *src
    ct.c_int64,                 # dstm_time_t time
    ct.c_int64,                 # dstm_time_t delta
]

libdstm.dstm_darray_get_rangev.restype = ct.c_int
libdstm.dstm_darray_get_rangev.argtypes = [
    ct.c_void_p,                # dstm_darray_t *darray
    ct.POINTER(ct.c_size_t),    # const size_t *indices
    ct.c_size_t,                # size_t count
    ct.POINTER(ct.c_void_p),    # void *const *dsts
    ct.c_int64,                 # dstm_time_t time
    ct.c_int64,                 # dstm_time_t delta
]

libdstm.dstm_darray_put_rangev.restype = ct.c_int
libdstm.dstm_darray_put_rangev.argtypes = [
    ct.c_void_p,                # dstm_darray_t *darray
    ct.POINTER(ct.c_size_t),    # const size_t *indices
    ct.c_size_t,                # size_t count
    ct.POINTER(ct.c_void_p),    # const void *const *srcs
    ct.c_int64,                 # dstm_time_t time
    ct.c_int64,                 # dstm_time_t delta
]

libdstm.dstm_init.restype = ct.c_int
libdstm.dstm_init.argtypes = [
    ct.POINTER(ct.c_int),
    ct.POINTER(ct.POINTER(ct.c_char_p))
    # ct.c_void_p,
    # ct.c_void_p
]

libdstm.dstm_finalize.restype = ct.c_int
libdstm.dstm_finalize.argtypes = []

libdstm.dstm_fini.restype = ct.c_int
libdstm.dstm_fini.argtypes = [ct.c_int64]

libdstm.dstm_log_setlevel.restype = ct.c_int
libdstm.dstm_log_setlevel.argtypes = [ct.c_int]

libdstm.dstm_distance_create_ring.restype = ct.c_int
libdstm.dstm_distance_create_ring.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]

libdstm.dstm_distance_get.restype = ct.c_int
libdstm.dstm_distance_get.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]

libdstm.dstm_distance_create_complete.restype = ct.c_int
libdstm.dstm_distance_create_complete.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]

libdstm.dstm_distance_destroy.restype = None
libdstm.dstm_distance_destroy.argtypes = [ct.POINTER(ct.c_void_p)]

libdstm.dstm_start.restype = ct.c_int
libdstm.dstm_start.argtypes = [ct.c_void_p]


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
    """Raise RuntimeError if ``rc`` is non-zero, labeled with ``name``."""
    if rc != 0:
        raise RuntimeError(f"{name} failed with code {rc}")


def darray_create(dtype, local_elements):
    """Create a distributed array with ``local_elements`` elements on this rank.

    Collective across the session. Element size is taken from ``dtype``; the
    global array is every rank's local block concatenated in rank order.

    Args:
        dtype: numpy dtype (or dtype-like); its itemsize is the C element_size.
        local_elements: number of elements owned by the calling rank.

    Returns:
        Darray: handle wrapping the new distributed array and its dtype.

    Raises:
        RuntimeError: on a non-zero return code from the C call.
    """
    dtype = np.dtype(dtype)
    handle = ct.POINTER(_Darray)()
    rc = libdstm.dstm_darray_create(ct.byref(handle), dtype.itemsize, local_elements)
    _check(rc, "dstm_darray_create")
    darray = Darray(handle, dtype)
    return darray


def darray_put(darray, global_index, value, time, delta):
    """Write a single element at ``global_index``, effective at time + delta.

    Args:
        darray: the Darray to write to.
        global_index: global element index to write.
        value: element data, coerced to a contiguous ``darray.dtype`` buffer.
        time: base logical timestamp.
        delta: non-negative offset added to ``time``; ``delta == 0`` requires
            the index be owned by the local rank.

    Raises:
        RuntimeError: on a non-zero return code (negative delta, or delta == 0
            for a remote index).
    """
    arr = np.ascontiguousarray(value, dtype=darray.dtype)
    rc = libdstm.dstm_darray_put(
        darray.handle, global_index, arr.ctypes.data, time, delta
    )
    _check(rc, "dstm_darray_put")


def darray_get(darray, global_index, dst, time, delta):
    """Read a single element at ``global_index``, effective at time + delta.

    Args:
        darray: the Darray to read from.
        global_index: global element index to read.
        dst: pre-allocated numpy array of ``darray.dtype`` whose buffer
            receives the element.
        time: base logical timestamp.
        delta: non-positive offset added to ``time``; ``delta == 0`` requires
            the index be owned by the local rank.

    Raises:
        RuntimeError: on a non-zero return code (positive delta, or delta == 0
            for a remote index).
    """
    rc = libdstm.dstm_darray_get(
        darray.handle, global_index, dst.ctypes.data, time, delta
    )
    _check(rc, "dstm_darray_get")


def darray_allgatherv(darray, src, dst, time, deltas):
    """All-to-all gather of every rank's local slice into ``dst``.

    Each rank publishes its slice with a put at ``time + deltas[self]`` and
    reads every other rank's slice with a get at ``time + deltas[r]``.

    Args:
        darray: the Darray to gather.
        src: local slice, coerced to a contiguous ``darray.dtype`` array.
        dst: pre-allocated contiguous numpy array of ``darray.dtype`` sized for
            the full global array.
        time: base logical timestamp shared by all participants.
        deltas: exactly ``world_size`` signed offsets; ``deltas[self]`` > 0 and
            every other entry < 0 (no zeros).

    Raises:
        TypeError: if ``dst`` is not a suitable contiguous array.
        RuntimeError: on a non-zero return code from the C call.
    """
    src = np.ascontiguousarray(src, dtype=darray.dtype)
    if not (isinstance(dst, np.ndarray) and dst.dtype == darray.dtype and dst.flags.c_contiguous):
        raise TypeError("dst must be a contiguous numpy array with darray.dtype")
    deltas = np.ascontiguousarray(deltas, dtype=np.int64)
    rc = libdstm.dstm_darray_allgatherv(
        darray.handle,
        src.ctypes.data,
        dst.ctypes.data,
        time,
        deltas.ctypes.data_as(ct.POINTER(ct.c_int64)),
    )
    _check(rc, "dstm_darray_allgatherv")


def _indices_ptr(indices):
    """Coerce ``indices`` to a contiguous size_t buffer; return (ptr, count)."""
    # data_as keeps the array alive via ptr._arr, so we need not retain idx
    idx = np.ascontiguousarray(indices, dtype=np.uintp)
    return idx.ctypes.data_as(ct.POINTER(ct.c_size_t)), len(idx)


def darray_get_range(darray, indices, dst, time, delta):
    """Read multiple global indices into one contiguous buffer.

    Args:
        darray: the Darray to read from.
        indices: sequence of global indices to read (any order; coerced to
            a contiguous uintp array). Its length is the element count.
        dst: pre-allocated contiguous numpy array of ``darray.dtype`` with at
            least ``len(indices)`` elements; element i is written to dst[i].
        time: base logical timestamp.
        delta: non-positive offset added to ``time`` (read constraint);
            ``delta == 0`` requires every index be locally owned.

    Raises:
        TypeError: if ``dst`` is not a suitable contiguous array.
        RuntimeError: on a non-zero return code from the C call.
    """
    idx_ptr, count = _indices_ptr(indices)
    if not (isinstance(dst, np.ndarray) and dst.dtype == darray.dtype
            and dst.flags.c_contiguous and dst.size >= count):
        raise TypeError("dst must be a contiguous numpy array of darray.dtype "
                        "with at least len(indices) elements")
    rc = libdstm.dstm_darray_get_range(
        darray.handle, idx_ptr, count, dst.ctypes.data, time, delta
    )
    _check(rc, "dstm_darray_get_range")


def darray_put_range(darray, indices, src, time, delta):
    """Write one contiguous buffer to multiple global indices.

    Args:
        darray: the Darray to write to.
        indices: sequence of global indices to write (coerced to a contiguous
            uintp array). Its length is the element count.
        src: source data coerced to a contiguous ``darray.dtype`` array;
            element i (src[i]) is written to ``indices[i]``.
        time: base logical timestamp.
        delta: non-negative offset added to ``time`` (write constraint);
            ``delta == 0`` requires every index be locally owned.

    Raises:
        RuntimeError: on a non-zero return code from the C call.
    """
    idx_ptr, count = _indices_ptr(indices)
    arr = np.ascontiguousarray(src, dtype=darray.dtype)
    rc = libdstm.dstm_darray_put_range(
        darray.handle, idx_ptr, count, arr.ctypes.data, time, delta
    )
    _check(rc, "dstm_darray_put_range")


def darray_get_rangev(darray, indices, dsts, time, delta):
    """Scattered-buffer variant of darray_get_range.

    Args:
        darray: the Darray to read from.
        indices: sequence of global indices to read. Its length is the count.
        dsts: list of ``len(indices)`` contiguous numpy arrays (each at least
            one ``darray.dtype`` element); index i is written into ``dsts[i]``.
        time: base logical timestamp.
        delta: non-positive offset added to ``time`` (read constraint);
            ``delta == 0`` requires every index be locally owned.

    Raises:
        RuntimeError: on a non-zero return code from the C call.
    """
    idx_ptr, count = _indices_ptr(indices)
    # keep the destination arrays referenced for the duration of the call
    ptr_arr = (ct.c_void_p * count)(*[d.ctypes.data for d in dsts])
    rc = libdstm.dstm_darray_get_rangev(
        darray.handle, idx_ptr, count,
        ct.cast(ptr_arr, ct.POINTER(ct.c_void_p)), time, delta
    )
    _check(rc, "dstm_darray_get_rangev")


def darray_put_rangev(darray, indices, srcs, time, delta):
    """Scattered-buffer variant of darray_put_range.

    Args:
        darray: the Darray to write to.
        indices: sequence of global indices to write. Its length is the count.
        srcs: list of ``len(indices)`` array-likes (each coerced to a
            contiguous ``darray.dtype`` buffer); ``srcs[i]`` is written to
            ``indices[i]``.
        time: base logical timestamp.
        delta: non-negative offset added to ``time`` (write constraint);
            ``delta == 0`` requires every index be locally owned.

    Raises:
        RuntimeError: on a non-zero return code from the C call.
    """
    idx_ptr, count = _indices_ptr(indices)
    # keep the coerced source buffers alive for the duration of the call
    bufs = [np.ascontiguousarray(s, dtype=darray.dtype) for s in srcs]
    ptr_arr = (ct.c_void_p * count)(*[b.ctypes.data for b in bufs])
    rc = libdstm.dstm_darray_put_rangev(
        darray.handle, idx_ptr, count,
        ct.cast(ptr_arr, ct.POINTER(ct.c_void_p)), time, delta
    )
    _check(rc, "dstm_darray_put_rangev")


def init(argv=None):
    """Initialize the dstm library (call once, after MPI is up).

    Args:
        argv: currently ignored; the C library is invoked with no args.

    Raises:
        RuntimeError: on a non-zero return code from the C call.
    """
    # if argv is None:
    #     argv = sys.argv
    # encoded = [s.encode() for s in argv]
    # argv_arr = (ct.c_char_p * (len(encoded) + 1))(*encoded, None)
    # argv_pp = ct.cast(argv_arr, ct.POINTER(ct.c_char_p))
    # argc = ct.c_int(len(encoded))
    # rc = libdstm.dstm_init(ct.byref(argc), ct.byref(argv_pp))
    rc = libdstm.dstm_init(None, None)
    _check(rc, "dstm_init")


def finalize():
    """Terminate the dstm library; call before the program exits.

    Raises:
        RuntimeError: on a non-zero return code from the C call.
    """
    rc = libdstm.dstm_finalize()
    _check(rc, "dstm_finalize")


def fini(last_time: int):
    """Flush all in-flight requests and shut down the local region.

    Args:
        last_time: highest effective timestep submitted by this rank, i.e.
            ``max(time + delta)`` over all put/get calls. Too small a value
            stalls the drain loop indefinitely.

    Raises:
        RuntimeError: on a non-zero return code from the C call.
    """
    rc = libdstm.dstm_fini(last_time)
    _check(rc, "dstm_fini")


def log_setlevel(level):
    """Set dstm log verbosity.

    Args:
        level: one of the module ``LOG_*`` constants (LOG_QUIET .. LOG_DEBUG).

    Raises:
        RuntimeError: on a non-zero return code from the C call.
    """
    rc = libdstm.dstm_log_setlevel(level)
    _check(rc, "dstm_log_setlevel")


def distance_get(dist: Distance, i: int, j: int) -> int:
    """Return the distance (hop count) between ranks ``i`` and ``j``.

    Args:
        dist: a Distance from one of the ``distance_create_*`` helpers.
        i: source rank index.
        j: destination rank index.

    Returns:
        int: the distance between rank ``i`` and rank ``j``.
    """
    distance = libdstm.dstm_distance_get(dist.handle, i, j)
    return distance


def distance_create_ring(size):
    """Create a ring-topology distance matrix.

    Each node connects to two neighbors in a circle; distance is the minimum
    hop count around the ring.

    Args:
        size: number of nodes (>= 3 for a valid ring).

    Returns:
        Distance: handle wrapping the new distance matrix.

    Raises:
        RuntimeError: on a non-zero return code from the C call.
    """
    handle = ct.c_void_p()
    rc = libdstm.dstm_distance_create_ring(ct.byref(handle), size)
    _check(rc, "dstm_distance_create_ring")
    return Distance(handle)


def distance_create_complete(size):
    """Create a fully-connected (complete graph) distance matrix.

    Every node links directly to every other; all off-diagonal distances are 1.

    Args:
        size: number of nodes (>= 1).

    Returns:
        Distance: handle wrapping the new distance matrix.

    Raises:
        RuntimeError: on a non-zero return code from the C call.
    """
    handle = ct.c_void_p()
    rc = libdstm.dstm_distance_create_complete(ct.byref(handle), size)
    _check(rc, "dstm_distance_create_complete")
    return Distance(handle)


def distance_destroy(dist):
    """Destroy a Distance and free its resources.

    Args:
        dist: the Distance to destroy; its handle is freed in place.
    """
    libdstm.dstm_distance_destroy(ct.byref(dist.handle))


def start(dist):
    """Start a dstm session using the given topology.

    Args:
        dist: a Distance describing the rank topology / lookahead.

    Raises:
        RuntimeError: on a non-zero return code from the C call.
    """
    rc = libdstm.dstm_start(dist.handle)
    _check(rc, "dstm_start")
