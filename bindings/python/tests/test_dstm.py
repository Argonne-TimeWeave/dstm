# Copyright 2019 UChicago Argonne, LLC.
# (c.f. AUTHORS, LICENSE)
#
# This file is part of the libnrm project.
# For more info, see https://github.com/anlsys/libnrm
#
# SPDX-License-Identifier: BSD-3-Clause
import os
os.environ["LIBDSTM_SO_"] = "/home/nick/Documents/repos/dstm/build/libdstm.so"

from mpi4py import MPI
import numpy as np
import pytest

import dstm


def test_lifecycle():
    world_size = MPI.COMM_WORLD.Get_size()
    dstm.init()
    dist = dstm.distance_create_complete(world_size)
    dstm.start(dist)
    dstm.fini(1)
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

    dstm.fini(1)
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

    dstm.fini(1)
    dstm.distance_destroy(dist)
    dstm.finalize()


def test_darray_allgather():
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    if world_size != 2:
        pytest.fail("World size must be 2")
    dstm.init()
    dstm.log_setlevel(dstm.LOG_ERROR)

    dist = dstm.distance_create_complete(world_size)
    dstm.start(dist)

    local_elements = [4, 2]
    darray = dstm.darray_create(np.int64, local_elements[rank])
    data = [np.array([1, 2, 3, 4], dtype=np.int64), np.array([5, 6], dtype=np.int64)]
    dst = np.zeros((darray.n_total_elements,), dtype=darray.dtype)

    for t in range(3):
        for i in range(data[rank].shape[0]):
            dstm.darray_put(darray, darray.rank_offsets[rank] + i, data[rank][i], t, 1)


    deltas = [np.array([1, -1], dtype=np.int64), np.array([-1, 1], dtype=np.int64)]
    src = data[rank] + 1
    dstm.darray_allgatherv(darray, src, dst, 2, deltas[rank])
    print(rank, dst)
    dstm.darray_allgatherv(darray, data[rank] * 2, dst, 3, deltas[rank])
    print(rank, dst)
    dstm.darray_allgatherv(darray, data[rank] * 3, dst, 4, deltas[rank])
    dstm.darray_allgatherv(darray, data[rank] * 4, dst, 5, deltas[rank])
    print(rank, dst)
 
    # ?? Changes in src, don't seem to be reflected in dst on other ranks.
    # rank 1 dst is [1 2 3 4 0 0]
    # rank 0 dst is [0 0 0 0 5 6]
    
    # fini should take the
    # last timestamp
    dstm.fini(6)
    dstm.distance_destroy(dist)
    dstm.finalize()
