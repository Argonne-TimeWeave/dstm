#!/usr/bin/env bats
# vim: set ft=bash:

bats_require_minimum_version 1.5.0

@test "mpirun -n 2 allgatherv" {
	run -0 --separate-stderr timeout --preserve-status 5 $LOG_COMPILER $LOG_FLAGS mpirun -n 2 $ABS_TOP_BUILDDIR/tests/mpi/allgatherv
}

@test "mpirun -n 4 allgatherv" {
	run -0 --separate-stderr timeout --preserve-status 5 $LOG_COMPILER $LOG_FLAGS mpirun -n 4 $ABS_TOP_BUILDDIR/tests/mpi/allgatherv
}

@test "mpirun -n 2 darray" {
	run -0 --separate-stderr timeout --preserve-status 5 $LOG_COMPILER $LOG_FLAGS mpirun -n 2 $ABS_TOP_BUILDDIR/tests/mpi/darray
}

@test "mpirun -n 4 darray" {
	run -0 --separate-stderr timeout --preserve-status 5 $LOG_COMPILER $LOG_FLAGS mpirun -n 4 $ABS_TOP_BUILDDIR/tests/mpi/darray
}

@test "mpirun -n 4 distances" {
	run -0 --separate-stderr timeout --preserve-status 5 $LOG_COMPILER $LOG_FLAGS mpirun -n 4 $ABS_TOP_BUILDDIR/tests/mpi/distances
}

@test "mpirun -n 8 distances" {
	run -0 --separate-stderr timeout --preserve-status 5 $LOG_COMPILER $LOG_FLAGS mpirun -n 8 $ABS_TOP_BUILDDIR/tests/mpi/distances
}

@test "mpirun -n 2 msg" {
	run -0 --separate-stderr timeout --preserve-status 5 $LOG_COMPILER $LOG_FLAGS mpirun -n 2 $ABS_TOP_BUILDDIR/tests/mpi/msg
}
