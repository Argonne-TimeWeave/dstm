# how to build on sst JLSE

## python3.9 or higher required

```
source ~/venv/dstm/bin/activate

cd ~/sst-core
rm -rf build
mkdir build
cd build

../sst-core-src/configure \
  --prefix="$HOME/sst" \
  --with-python=/usr/bin/python3.12-config

make -j"$(nproc)"
make install


export PATH="$HOME/sst/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/sst/lib:$LD_LIBRARY_PATH"
sst --version
```

## building this sst benchmark

```
make
make register
```

## running it

```
SST_MODEL_OPTIONS="--racks 16 --domains 4 --sst-ranks 4 --steps 300 --jobs jobs.csv" \
mpirun -n 4 sst run.py
```
