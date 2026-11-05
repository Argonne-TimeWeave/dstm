{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  env.CC = "mpicc";
  env.CFLAGS = 
    "-std=c17 -Wall -Wextra -Werror -Wno-builtin-declaration-mismatch";

  devcontainer.enable = true;

  packages = with pkgs; [
    # builddeps
    pkg-config
    automake
    autoconf
    libtool

    # deps
    openmpi
    check
    jansson
    bats

    # dev
    gdb
    valgrind
    git
  ];

  # tooling
  languages.c.enable = true;
  languages.python.enable = true;
}
