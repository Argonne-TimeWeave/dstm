# latexmk configuration for the DSTM specification.

# pdflatex (not xe/lua): the document uses only standard TeX fonts, and
# pdflatex keeps the build fast enough for `make watch` to be usable.
$pdf_mode = 1;

$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';

# Keep every intermediate artifact out of the source tree; spec/build/ is
# gitignored wholesale, so `git status` stays clean while editing.
$out_dir = 'build';

# pdflatex writes the .aux of an \include'd file next to it, mirroring the
# source layout under $out_dir -- but it will not create the intermediate
# directory itself and dies with "Emergency stop" instead. Pre-create the
# mirror so \include{sections/...} works on a clean checkout, for both
# `make` and `make watch`.
use File::Path qw(make_path);
make_path("$out_dir/sections");

# biblatex backend.
$bibtex_use = 2;

# latexmk does not know about these by default.
$clean_ext = 'synctex.gz run.xml bbl bcf nav snm vrb';
