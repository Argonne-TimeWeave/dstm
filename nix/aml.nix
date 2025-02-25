{ stdenv
, autoreconfHook
, git
, pkgconf
, numactl
, hwloc
}:
stdenv.mkDerivation {
  src = fetchGit {
    url = "https://github.com/anlsys/aml.git";
    ref = "master";
    submodules = true;
  };
  name = "aml";
  prePatch = "echo 0.2.0 > .tarball-version";
  nativeBuildInputs = [ autoreconfHook pkgconf git ];
  propagatedBuildInputs = [ hwloc numactl ];
}
