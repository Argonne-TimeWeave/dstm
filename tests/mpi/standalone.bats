#!/usr/bin/env bats
# vim: set ft=bash:

bats_require_minimum_version 1.5.0

@test "mpirun -n 2 allgatherv" {
	run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 2 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/allgatherv
}

@test "mpirun -n 4 allgatherv" {
	run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 4 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/allgatherv
}

@test "mpirun -n 2 darray" {
	run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 2 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/darray
}

@test "mpirun -n 4 darray" {
	run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 4 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/darray
}

@test "mpirun -n 4 distances" {
	run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 4 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/distances
}

@test "mpirun -n 8 distances" {
	run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 8 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/distances
}

@test "mpirun -n 2 msg" {
	run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 2 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/msg
}

@test "mpirun -n 2 get_or_default" {
	run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 2 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/get_or_default
}

@test "mpirun -n 4 get_or_default" {
	run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 4 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/get_or_default
}

@test "mpirun -n 2 local_delta" {
	run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 2 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/local_delta
}

@test "mpirun -n 4 local_delta" {
	run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 4 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/local_delta
}

@test "mpirun -n 2 multi_session" {
	run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 2 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/multi_session
}

@test "mpirun -n 4 multi_session" {
	run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 4 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/multi_session
}

# put_after_get is a NEGATIVE test: it deliberately violates the
# put-before-get phase rule and must abort.  We assert the mpirun exits
# with non-zero status and stderr contains the contract-violation
# marker logged by dstm_request_progress_local_put.
@test "mpirun -n 2 put_after_get aborts on violation" {
	run --separate-stderr timeout --preserve-status 5 mpirun -n 2 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/put_after_get
	[ "$status" -ne 0 ]
	[[ "$stderr" == *"contract violation"* ]]
}

@test "mpirun -n 4 put_after_get aborts on violation" {
	run --separate-stderr timeout --preserve-status 5 mpirun -n 4 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/put_after_get
	[ "$status" -ne 0 ]
	[[ "$stderr" == *"contract violation"* ]]
}

@test "mpirun -n 2 init_comm_world" {
        run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 2 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/init_comm_world
}

@test "mpirun -n 4 init_comm" {
        run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 4 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/init_comm
}

@test "mpirun -n 8 init_comm" {
        run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 8 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/init_comm
}

@test "mpirun -n 2 dgraph" {
        run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 2 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/dgraph
}

@test "mpirun -n 4 dgraph" {
        run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 4 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/dgraph
}

@test "mpirun -n 4 mini_airport" {
        run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 4 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/mini_airport
}

@test "mpirun -n 2 multi_region" {
        run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 2 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/multi_region
}

@test "mpirun -n 4 multi_region" {
        run -0 --separate-stderr timeout --preserve-status 5 mpirun -n 4 $LOG_COMPILER $LOG_FLAGS $ABS_TOP_BUILDDIR/tests/mpi/multi_region
}
