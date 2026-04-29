{ stdenv
, autoreconfHook
, check
, git
, jansson
, pkg-config
, mpich
}:
stdenv.mkDerivation {
  src = ../.;
  name = "dstm";
  nativeBuildInputs = [ autoreconfHook pkg-config git ];
  buildInputs = [ check jansson mpich ];
}
