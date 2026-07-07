# ROSS airport sim

ROSS is a prior work in the space time/Time-Warp space. They have a model of airports in their github organization: https://github.com/ROSS-org/ROSS-Models/tree/master/airport

The code is here for reference (since i adjusted it to be more comparable to the existing DSTM version). in order to build you will need to first build ROSS and then simlink or copy this code into the related ROSS repo.

## build

```
# build ROSS first
git clone http://github.com/ross-org/ROSS.git
mkdir -p ~/ROSS-build
cd ~/ROSS-build
cmake ~/ROSS -DROSS_BUILD_MODELS=ON \
    -DCMAKE_INSTALL_PREFIX="$(realpath ./bin)" \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_FLAGS="-g -Wall"
make -j 12

cd ~/

# then grab the models repo
git clone git@github.com:ROSS-org/ROSS-Models.git

# symlink airport into ROSS/models
ln -s ~/ROSS-Models/airport ~/ROSS/models/airport

cd ~/ROSS-build   

# Re-run cmake so it picks up the new subdirectory
cmake .

# Build just the airport model
make airport

cd ~/ROSS-build/models/airport
```

Somewhere in the middle or afterwards you can copy the airport files here into that directory to build again. 
