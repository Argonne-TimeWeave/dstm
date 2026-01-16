#!/usr/bin/env bats
# vim: set ft=bash:

bats_require_minimum_version 1.5.0

@test "mpirun -n 2 pdes" {
	run -0 --separate-stderr $LOG_COMPILER $LOG_FLAGS mpirun -n 2 $ABS_TOP_BUILDDIR/tests/mpi/pdes
}
