# Copyright 2019 UChicago Argonne, LLC.
# (c.f. AUTHORS, LICENSE)
#
# This file is part of the libnrm project.
# For more info, see https://github.com/anlsys/libnrm
#
# SPDX-License-Identifier: BSD-3-Clause
import os
os.environ["LIBDSTM_SO_"] = f"{os.path.dirname(os.path.abspath(__file__))}/../../../.libs/libdstm.so"

from mpi4py import MPI
import numpy as np
import pytest

import dstm


def test_lifecycle():
    world_size = MPI.COMM_WORLD.Get_size()
    dstm.init()
    dstm.log_setlevel(dstm.LOG_DEBUG)
    dist = dstm.distance_create_complete(world_size)
    dstm.start(dist)
    dstm.fini(0)
    dstm.distance_destroy(dist)
    dstm.finalize()


def test_darray_simple():
    world_size = MPI.COMM_WORLD.Get_size()
    dstm.init()
    dstm.log_setlevel(dstm.LOG_DEBUG)
    
    dist = dstm.distance_create_complete(world_size)
    dstm.start(dist)

    darray = dstm.darray_create(np.int64, 1)
    assert darray.element_size == np.dtype(np.int64).itemsize
    assert darray.n_total_elements == 2
    assert darray.n_local_elements == 1
    assert darray.rank_offsets[0] == 0
    assert darray.rank_offsets[1] == 1
    assert darray.rank_n_elements[0] == 1
    assert darray.rank_n_elements[1] == 1

    dstm.fini(0)
    dstm.distance_destroy(dist)
    dstm.finalize()


def test_darray_non_uniform():
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    if world_size != 2:
        pytest.fail("World size must be 2")
    dstm.init()
    dstm.log_setlevel(dstm.LOG_DEBUG)

    dist = dstm.distance_create_complete(world_size)
    dstm.start(dist)

    local_elements = [4, 2]
    darray = dstm.darray_create(np.int64, local_elements[rank])
  
    assert darray.element_size == np.dtype(np.int64).itemsize
    assert darray.n_total_elements == sum(local_elements)
    assert darray.n_local_elements == local_elements[rank]
    assert darray.rank_offsets[0] == 0
    assert darray.rank_offsets[1] == 4
    assert darray.rank_n_elements[0] == 4
    assert darray.rank_n_elements[1] == 2

    dstm.fini(0)
    dstm.distance_destroy(dist)
    dstm.finalize()


def test_darray_range_get_put():
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    dstm.init()
    dstm.log_setlevel(dstm.LOG_ERROR)

    dist = dstm.distance_create_complete(world_size)
    dstm.start(dist)

    per_rank = 6
    t_put, t_get, tag = 0, 2, 4
    # creates an array with 6 element for each rank
    darray = dstm.darray_create(np.int64, per_rank)

    # operate on this rank's own contiguous partition
    off = darray.rank_offsets[rank]
    indices = np.array([off + i for i in range(per_rank)], dtype=np.uintp)
    payload = np.array([(off + i) * 100 + tag for i in range(per_rank)],
                       dtype=np.int64)
    
    # put takes effect at t_put + 1; get at t_get - 1 reads that same time
    dstm.darray_put_range(darray, indices, payload, t_put, 1)

    get_buff = np.full(per_rank, -1, dtype=np.int64)
    dstm.darray_get_range(darray, indices, get_buff, t_get, -1)
    assert np.array_equal(get_buff, payload)

    get_buff = np.full(2, -1, dtype=darray.dtype)
    if rank == 0:
        off = darray.rank_offsets[1]
    else: 
        off = darray.rank_offsets[0]

    # get first 3 elements from neighbor
    indices = np.array([off + i for i in range(3)])
    get_buff = np.full(3, -1, dtype=np.int64)
    dstm.darray_get_range(darray, indices, get_buff, t_get, -1)
    # expected 3 elements from neighbor
    exp = np.array([(off + i) * 100 + tag for i in range(3)], dtype=darray.dtype)
    assert np.array_equal(get_buff, exp)

    dstm.fini(t_put + 1)
    dstm.distance_destroy(dist)
    dstm.finalize()


def test_darray_rangev_get_put():
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    dstm.init()
    dstm.log_setlevel(dstm.LOG_ERROR)

    dist = dstm.distance_create_complete(world_size)
    dstm.start(dist)

    per_rank = 5
    t_put, t_get, tag = 0, 2, 5
    darray = dstm.darray_create(np.int64, per_rank)

    off = darray.rank_offsets[rank]
    indices = np.array([off + i for i in range(per_rank)], dtype=np.uintp)
    # scattered per-element buffers exercise the per-element pointer indirection
    srcs = [np.array([(off + i) * 1000 + tag], dtype=np.int64)
            for i in range(per_rank)]
    dsts = [np.full(1, -1, dtype=np.int64) for _ in range(per_rank)]

    dstm.darray_put_rangev(darray, indices, srcs, t_put, 1)
    dstm.darray_get_rangev(darray, indices, dsts, t_get, -1)

    for s, d in zip(srcs, dsts):
        assert d[0] == s[0]

    dstm.fini(t_put + 1)
    dstm.distance_destroy(dist)
    dstm.finalize()
