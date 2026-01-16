# development shell, includes aml dependencies and dev-related helpers
{ pkgs ? import ./. { } }:
with pkgs;
mkShell {
  inputsFrom = [ dstm ];
  nativeBuildInputs = [ autoreconfHook pkg-config ];
  buildInputs = [
    # deps for debug
    gdb
    valgrind
    # style checks
    clang-tools
    python3
    llvmPackages.clang-unwrapped.python
    bear
    # python bindings
    python3Packages.flake8
    python3Packages.black
    # extra tools for testing
    jq
    bats
  ];
  CFLAGS =
    "-std=c17 -Wall -Wextra -Werror -Wno-builtin-declaration-mismatch";
}
