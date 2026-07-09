EXTRA_DIST = \
	tests/__init__.py \
	tests/test_dstm.py \
	pyproject.toml \
	setup.py

pkgpythondir = ${pythondir}/dstm
pkgpython_PYTHON = dstm/__init__.py \
		   dstm/dstm.py

check: FORCE
	ABS_TOP_BUILDDIR=$(abs_top_builddir) PYTHONPATH=$(srcdir)/:$(PYTHONPATH) LIBDSTM_SO_=$(top_builddir)/.libs/libdstm.so mpirun -n 2 $(PYTHON) -m pytest --with-mpi

FORCE:
