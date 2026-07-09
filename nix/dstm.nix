{ stdenv
, autoreconfHook
, check
, git
, jansson
, pkg-config
, openmpi
}:
stdenv.mkDerivation {
  src = ../.;
  name = "dstm";
  nativeBuildInputs = [ autoreconfHook pkg-config git ];
  buildInputs = [ check jansson openmpi ];
}
