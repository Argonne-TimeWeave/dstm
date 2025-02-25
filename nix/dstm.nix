{ stdenv
, aml
, argobots
, autoreconfHook
, check
, git
, pkg-config
}:
stdenv.mkDerivation {
  src = ../.;
  name = "dstm";
  nativeBuildInputs = [ autoreconfHook pkg-config git ];
  buildInputs = [ aml argobots check ];
}
