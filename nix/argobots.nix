{ stdenv
, autoreconfHook
, git
, pkg-config
, hwloc
}:
stdenv.mkDerivation {
  src = fetchGit {
    url = "https://github.com/pmodels/argobots.git";
    ref = "main";
  };
  name = "argobots";
  nativeBuildInputs = [ autoreconfHook pkg-config git ];
  buildInputs = [ hwloc ];
}
