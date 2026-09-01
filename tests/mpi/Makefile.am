AM_CPPFLAGS = \
	      -include $(top_builddir)/src/config.h \
	      -I$(top_srcdir)/include \
	      -I$(top_builddir)/include

AM_LDFLAGS = $(AML_LIBS)
AM_CFLAGS = $(AML_CFLAGS)

LDADD = $(top_builddir)/libdstm.la

AM_COLOR_TESTS = yes

#VALGRIND_SUPPRESSIONS_FILES = $(top_srcdir)/.valgrind.supp

# valgrind support
@VALGRIND_CHECK_RULES@

# can't run those tests in parallel, they do mpirun
.NOTPARALLEL:

COMPILED_TESTS = \
		allgatherv \
		darray \
		distances \
		get_or_default \
               init_comm \
               init_comm_world \
		local_delta \
		msg \
		multi_session \
        multi_region \
        dgraph \
        mini_airport \
		put_after_get

check_PROGRAMS = $(COMPILED_TESTS)

BATS_TESTS = \
	     standalone.bats

TEST_EXTENSIONS = .bats
BATS_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' $(SHELL) \
		$(top_srcdir)/build-aux/tap-driver.sh
BATS_LOG_COMPILER = $(SHELL) $(top_builddir)/tests/mpi/bats-driver.sh

TESTS = $(BATS_TESTS)
EXTRA_DIST = $(BATS_TESTS)
