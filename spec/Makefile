# Build driver for the DSTM specification.
#
# Standalone on purpose: nothing here is wired into the project's autotools
# build, so a missing LaTeX toolchain can never break `make` at the top level.
#
#   make            build build/main.pdf
#   make watch      rebuild + refresh preview on every save
#   make view       build, then open the PDF
#   make clean      remove auxiliary files, keep the PDF
#   make distclean  remove build/ entirely
#   make pdf-manual build without latexmk (pdflatex/biber invoked by hand)

MAIN     := main
BUILDDIR := build

LATEXMK  := latexmk
PDFLATEX := pdflatex
BIBER    := biber

# -halt-on-error turns a silent 300-line log into an immediate failure.
PDFLATEX_FLAGS := -interaction=nonstopmode -halt-on-error -file-line-error

SOURCES := $(wildcard *.tex) $(wildcard sections/*.tex) refs.bib

.PHONY: all pdf watch view clean distclean pdf-manual help

all: pdf

re: clean pdf

pdf: $(BUILDDIR)/$(MAIN).pdf

$(BUILDDIR)/$(MAIN).pdf: $(SOURCES)
	$(LATEXMK) $(MAIN)

watch:
	$(LATEXMK) -pvc $(MAIN)

view: pdf
	$(LATEXMK) -pv $(MAIN)

clean:
	$(LATEXMK) -c

distclean:
	-$(LATEXMK) -C
	rm -rf $(BUILDDIR)

# Fallback for environments without latexmk. The fixed point of
# pdflatex/biber/pdflatex/pdflatex resolves biblatex citations plus the
# forward references produced by cleveref and the theorem counters.
pdf-manual:
	mkdir -p $(BUILDDIR)/sections
	$(PDFLATEX) $(PDFLATEX_FLAGS) -output-directory=$(BUILDDIR) $(MAIN).tex
	-cd $(BUILDDIR) && $(BIBER) $(MAIN)
	$(PDFLATEX) $(PDFLATEX_FLAGS) -output-directory=$(BUILDDIR) $(MAIN).tex
	$(PDFLATEX) $(PDFLATEX_FLAGS) -output-directory=$(BUILDDIR) $(MAIN).tex

help:
	@echo "targets: pdf (default), watch, view, clean, distclean, pdf-manual"
