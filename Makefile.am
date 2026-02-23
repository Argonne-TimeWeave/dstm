bin_PROGRAMS = 
check_PROGRAMS = 
TESTS = 
SUBDIRS = 
BUILT_SOURCES = 
EXTRA_DIST = 
CLEANFILES = 
ACLOCAL_AMFLAGS = -I m4

AM_CPPFLAGS = \
	      -include $(top_builddir)/src/config.h \
	      -I$(top_srcdir)/include \
	      -I$(top_builddir)/include

AM_CFLAGS = $(AML_CFLAGS)

AM_LDFLAGS = $(AML_LIBS)

EXTRA_DIST += autogen.sh dstm.pc README.markdown LICENSE AUTHORS

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = dstm.pc

###############################################################################
# Version management
###############################################################################

EXTRA_DIST += $(top_srcdir)/.version
BUILT_SOURCES += $(top_srcdir)/.version
$(top_srcdir)/.version:
	echo $(VERSION) > $@-t && mv $@-t $@
dist-hook:
	echo $(VERSION) > $(distdir)/.tarball-version

###############################################################################
# Includes
###############################################################################

include_HEADERS = include/dstm.h

include_dstmdir=$(includedir)/dstm
include_dstm_HEADERS = \
		      include/dstm/version.h

noinst_include_internaldir = $(includedir)/internal
noinst_include_internal_HEADERS = \
				  include/internal/region.h
###############################################################################
# Library
###############################################################################

lib_LTLIBRARIES = libdstm.la

libdstm_la_SOURCES = \
		    src/log.c \
		    src/error.c \
		    src/region.c \
		    src/ops.c \
		    src/msg.c \
		    src/dstm.c \
		    src/darray.c

libdstm_la_LIBADD = $(LTLIBOBJS)

###############################################################################
# Bindings
###############################################################################

if HAVE_PYTHON
SUBDIRS += bindings/python
endif

###############################################################################
# TESTS
###############################################################################

# mpi tests use a different driver (bats), so we configure them in a subdir
SUBDIRS += tests/mpi

AM_COLOR_TESTS = yes
AM_CPPFLAGS += $(CHECK_CFLAGS)
AM_CFLAGS += $(CHECK_CFLAGS)
AM_LDFLAGS += $(CHECK_LIBS)
LDADD = libdstm.la

VALGRIND_SUPPRESSIONS_FILES = $(top_srcdir)/.valgrind.supp

# valgrind support
@VALGRIND_CHECK_RULES@

COMPILED_TESTS = \
		tests/core

# unit tests
UNIT_TESTS = \
	     $(COMPILED_TESTS)

check_PROGRAMS += $(COMPILED_TESTS)
TESTS += $(UNIT_TESTS)

