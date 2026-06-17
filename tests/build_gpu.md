# GPU memory test

## Overview

These tests verify that DSTM's buddy allocator works correctly with NVIDIA CUDA Unified Virtual Memory (UVM). By wrapping dstm\_region\_create\_with\_buffer() with a cudaMallocManaged buffer, we can allocate memory that is accessible from both CPU and GPU without explicit data transfers. The test suite validates region creation/destruction, allocation patterns, pointer conversions, and CPU-side memory access. GPU-side tests use CUDA kernels to verify true bidirectional coherence: CPU writes are visible to GPU kernels, and GPU kernel writes are visible to the CPU.

## build 

```
nvcc -x cu test_dstm_region_gpu.c -o test_dstm_region_gpu \
    -I$HOME/opt/check/include -I../include -I../include/internal \
    -L$HOME/opt/check/lib -L../.libs \
    -lcheck -ldstm -lpthread -lrt -lm

export LD_LIBRARY_PATH=$HOME/opt/check/lib:$HOME/dstm/.libs:$LD_LIBRARY_PATH
./test_dstm_region_gpu

if error:
CK_FORK=no LD_LIBRARY_PATH=$HOME/opt/check/lib:../.libs:$LD_LIBRARY_PATH ./test_dstm_region_gpu
```

## Ran on 06/17/2026:

```
Running GPU tests on: Tesla P100-PCIE-16GB
  Compute capability: 6.0
  Total memory: 15.89 GB
  Managed memory: supported

Running suite(s): DSTM GPU Region
100%: Checks: 10, Failures: 0, Errors: 0
```
