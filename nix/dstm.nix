{ stdenv
, autoreconfHook
, check
, git
, pkg-config
, mpich
}:
stdenv.mkDerivation {
  src = ../.;
  name = "dstm";
  nativeBuildInputs = [ autoreconfHook pkg-config git ];
  buildInputs = [ check mpich ];
}
