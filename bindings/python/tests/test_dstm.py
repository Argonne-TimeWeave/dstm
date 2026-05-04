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
    dstm.init()
    dstm.fini()
    dstm.finalize()


def test_darray_simple():
    world_size = MPI.COMM_WORLD.Get_size()
    dstm.init()
    dstm.log_setlevel(dstm.LOG_DEBUG)
    # TODO: NEED TO CREATE DISTANCE, AND CALL START
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

    dstm.fini()
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

    dstm.fini()
    dstm.distance_destroy(dist)
    dstm.finalize()


def test_darray_allgather():
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
    data = [np.array([1, 2, 3, 4], dtype=np.int64), np.array([5, 6], dtype=np.int64)]

    dst = np.empty((world_size,), dtype=darray.dtype)
    deltas = np.array([0, 0])
    dstm.darray_allgatherv(darray, data[rank], dst, 1, deltas)
    
    dstm.fini()
    dstm.distance_destroy(dist)
    dstm.finalize()
