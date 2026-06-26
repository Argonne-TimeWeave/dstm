AM_CPPFLAGS = \
	      -include $(top_builddir)/src/config.h \
	      -I$(top_srcdir)/include \
	      -I$(top_builddir)/include

AM_LDFLAGS = $(AML_LIBS)
AM_CFLAGS = $(AML_CFLAGS)

LDADD = $(top_builddir)/libdstm.la

noinst_PROGRAMS = zombies 

zombies_SOURCES = \
	zombie.c \
    zombie.h \
    event.h \
    human.h \
    main.c

EXTRA_DIST = README.md
