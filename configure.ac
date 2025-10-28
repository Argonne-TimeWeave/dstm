# see semver.org for version info
#################################

# for future use, once we have a tag for a version
#m4_define([VERSION_STRING], m4_esyscmd([build-aux/git-version-gen .tarball-version]))
m4_define([VERSION_STRING], [0.1.0.alpha])
m4_define([VERSION_FIELDS], m4_split(VERSION_STRING, [\.]))
m4_define([VERSION_MAJOR], m4_argn(1, VERSION_FIELDS))
m4_define([VERSION_MINOR], m4_argn(2, VERSION_FIELDS))
m4_define([VERSION_PATCH], m4_argn(3, VERSION_FIELDS))
m4_define([VERSION_REVISION], m4_argn(4, VERSION_FIELDS))

# init
######

AC_INIT([dstm],[0.1.0],[swann@anl.gov])
AC_CONFIG_SRCDIR([include/dstm.h])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])

# needed for tests
##################

AC_REQUIRE_AUX_FILE([git-version-gen])

# automake should fail on any error
AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects 1.12])

# check for programs
####################
AC_LANG([C])
AC_USE_SYSTEM_EXTENSIONS
AC_PROG_CC
AC_PROG_CC_C99
AC_PROG_CPP
AC_TYPE_SIZE_T
AC_TYPE_INTPTR_T
AM_PROG_AR
LT_INIT

# extra configuration
#####################

AC_SUBST([PACKAGE_VERSION_MAJOR],[VERSION_MAJOR])
AC_SUBST([PACKAGE_VERSION_MINOR],[VERSION_MINOR])
AC_SUBST([PACKAGE_VERSION_PATCH],[VERSION_PATCH])
AC_SUBST([PACKAGE_VERSION_REVISION],[VERSION_REVISION])

# support for testing with valgrind
###################################

AX_VALGRIND_DFLT([helgrind], [off])
AX_VALGRIND_DFLT([drd], [off])
AX_VALGRIND_DFLT([sgcheck], [off])
AX_VALGRIND_CHECK

# support for libtool
#####################
LT_INIT
AC_SEARCH_LIBS([dlsym], [dl dld], [], [
	AC_MSG_ERROR([unable to find the dlsym() function])
])

# dependencies
##############
PKG_CHECK_MODULES([CHECK],[check],,
		 AC_MSG_ERROR([unable to find check]))

PKG_CHECK_MODULES([AML],[aml],,
		  AC_MSG_ERROR([unable to find aml]))

# PYTHON
#########

AC_ARG_WITH([python],
	    [AS_HELP_STRING([--with-python],
			    [Install Python Client interface])],
	    [with_python=$withval], [with_python=no])
AS_IF([test "x$with_python" != "xno"],
      [
       AM_PATH_PYTHON([3.8], have_python=1, have_python=0)
      ],
      [have_python=0]
)

AM_CONDITIONAL([HAVE_PYTHON],[test "$have_python" == "1"])
AC_DEFINE_UNQUOTED([HAVE_PYTHON],[$have_python], [python support])
AC_SUBST([HAVE_PYTHON],[$have_python])

# options
#########

AC_CONFIG_HEADERS([src/config.h])

AC_CONFIG_FILES([Makefile
		 bindings/python/Makefile
		 dstm.pc
		 include/dstm/version.h])
AC_OUTPUT

# print out what was configured
cat <<EOF
-------------------------------------------------------------------------------
DSTM

Version: $PACKAGE_VERSION

FEATURES:
---------

CHECK:
=======

CFLAGS:  $CHECK_CFLAGS
LDFLAGS: $CHECK_LIBS

AML:
=======

CFLAGS:  $AML_CFLAGS
LDFLAGS: $AML_LIBS

PYTHON BINDINGS:
======

Active:  $have_python
PYTHON:  $PYTHON

-------------------------------------------------------------------------------
EOF
