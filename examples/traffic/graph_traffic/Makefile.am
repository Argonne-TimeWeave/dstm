AM_CPPFLAGS = \
	      -include $(top_builddir)/src/config.h \
	      -I$(top_srcdir)/include \
	      -I$(top_builddir)/include

AM_LDFLAGS = $(AML_LIBS)
AM_CFLAGS = $(AML_CFLAGS)

LDADD = $(top_builddir)/libdstm.la

noinst_PROGRAMS = traffic

traffic_SOURCES = \
	traffic.c \
	city.c \
	city.h \
    car.c \
    car.h \
    chan.c \
    chan.h 

EXTRA_DIST = README.md
