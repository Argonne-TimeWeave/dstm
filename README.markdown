A Runtime and Programming Model for Distributed Shared Timestamped Memory
=========================================================================

## Requirements

* autoconf
* automake
* libtool
* pkgconf
* AML
* Argobots

## Building in Nix

To compile this library using the provided nix-shell.

1. Install nix-shell 
   `curl -L https://nixos.org/nix/install | sh`
2. Grab the git-version-gen
   `curl -L https://git.savannah.gnu.org/cgit/gnulib.git/plain/build-aux/git-version-gen -o build-aux/git-version-gen`
3. Run `nix-shell` in this dir to launch a working dev environment 
4. Run `sh autogen.sh; ./configure; make`

## Installation

sh autogen.sh
./configure
make -j install
