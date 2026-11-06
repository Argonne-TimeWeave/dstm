/*******************************************************************************
 * test_dstm_region_gpu.c
 * 
 * Unit tests for DSTM region allocator using NVIDIA CUDA Unified Memory (UVM)
 * 
 * Compile with:
 *   nvcc -x cu test_dstm_region_gpu.c -o test_dstm_region_gpu \
 *        -lcheck -lpthread -lrt -lm -I/path/to/dstm/include -L/path/to/dstm -ldstm
 * 
 * Or with separate compilation:
 *   nvcc -c test_dstm_region_gpu.c -o test_dstm_region_gpu.o
 *   gcc test_dstm_region_gpu.o -o test_dstm_region_gpu \
 *       -lcheck -lpthread -lrt -lm -lcudart -ldstm
 ******************************************************************************/

#include <check.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Include your DSTM headers */
#include "dstm.h"
#include "region.h"

#ifdef __cplusplus
}
#endif

/*******************************************************************************
 * GPU Kernels for Memory Verification
 ******************************************************************************/

/* Kernel: Write a pattern from GPU */
__global__ void gpu_write_pattern(unsigned char *ptr, size_t size, unsigned char value)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        ptr[idx] = value;
    }
}

/* Kernel: Verify a pattern from GPU, set result to 0 if mismatch */
__global__ void gpu_verify_pattern(unsigned char *ptr, size_t size, 
                                    unsigned char expected, int *result)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (ptr[idx] != expected) {
            atomicExch(result, 0);  /* Set to 0 on mismatch */
        }
    }
}

/* Kernel: Increment each byte (read-modify-write from GPU) */
__global__ void gpu_increment_bytes(unsigned char *ptr, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        ptr[idx] += 1;
    }
}

/*******************************************************************************
 * GPU Region Creation Wrapper
 * 
 * This wraps dstm_region_create_with_buffer() to use CUDA Unified Memory
 ******************************************************************************/

typedef struct {
    dstm_region_t *region;
    void *gpu_buffer;        /* The cudaMallocManaged buffer */
    size_t allocated_size;   /* Actual allocated size (power of 2) */
} dstm_gpu_region_t;

/* Helper: round up to next power of 2 */
static size_t gpu_next_power_of_2(size_t x)
{
    if (x == 0)
        return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    x++;
    return x;
}

/**
 * Create a DSTM region backed by CUDA Unified Memory
 * 
 * @param gpu_region  Output: GPU region wrapper structure
 * @param size        Requested size (will be rounded up to power of 2)
 * @return            0 on success, negative error code on failure
 */
int dstm_region_create_gpu(dstm_gpu_region_t **gpu_region, size_t size)
{
    if (gpu_region == NULL || size == 0)
        return -1;

    /* Allocate wrapper structure */
    dstm_gpu_region_t *gr = (dstm_gpu_region_t *)calloc(1, sizeof(dstm_gpu_region_t));
    if (gr == NULL)
        return -1;

    /* Round size to power of 2 (required by buddy allocator) */
    gr->allocated_size = gpu_next_power_of_2(size);

    /* Allocate CUDA Unified Memory */
    cudaError_t cuda_err = cudaMallocManaged(&gr->gpu_buffer, 
                                              gr->allocated_size);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", 
                cudaGetErrorString(cuda_err));
        free(gr);
        return -1;
    }

    /* Initialize the buffer to zero */
    memset(gr->gpu_buffer, 0, gr->allocated_size);

    /* Create DSTM region with our GPU buffer */
    int dstm_err = dstm_region_create_with_buffer(&gr->region, 
                                                   gr->gpu_buffer, 
                                                   gr->allocated_size);
    if (dstm_err != DSTM_SUCCESS) {
        fprintf(stderr, "dstm_region_create_with_buffer failed: %d\n", 
                dstm_err);
        cudaFree(gr->gpu_buffer);
        free(gr);
        return dstm_err;
    }

    *gpu_region = gr;
    return DSTM_SUCCESS;
}

/**
 * Destroy a GPU-backed DSTM region
 */
void dstm_region_destroy_gpu(dstm_gpu_region_t **gpu_region)
{
    if (gpu_region == NULL || *gpu_region == NULL)
        return;

    dstm_gpu_region_t *gr = *gpu_region;

    /* Destroy the DSTM region first (this won't free our buffer since 
     * owns_buffer is 0 when using create_with_buffer) */
    if (gr->region != NULL)
        dstm_region_destroy(&gr->region);

    /* Now free the CUDA memory */
    if (gr->gpu_buffer != NULL) {
        cudaError_t err = cudaFree(gr->gpu_buffer);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaFree failed: %s\n", 
                    cudaGetErrorString(err));
        }
    }

    free(gr);
    *gpu_region = NULL;
}

/**
 * Synchronize GPU to ensure all operations are complete
 */
int dstm_region_gpu_sync(void)
{
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Prefetch region data to GPU
 */
int dstm_region_prefetch_to_gpu(dstm_gpu_region_t *gr, int device_id)
{
    if (gr == NULL || gr->gpu_buffer == NULL)
        return -1;

    cudaError_t err = cudaMemPrefetchAsync(gr->gpu_buffer, 
                                            gr->allocated_size, 
                                            device_id, 
                                            NULL);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Prefetch region data to CPU
 */
int dstm_region_prefetch_to_cpu(dstm_gpu_region_t *gr)
{
    if (gr == NULL || gr->gpu_buffer == NULL)
        return -1;

    cudaError_t err = cudaMemPrefetchAsync(gr->gpu_buffer, 
                                            gr->allocated_size, 
                                            cudaCpuDeviceId, 
                                            NULL);
    return (err == cudaSuccess) ? 0 : -1;
}

/*******************************************************************************
 * Test Helpers
 ******************************************************************************/

/* Launch config helper */
static void get_launch_config(size_t size, int *blocks, int *threads)
{
    *threads = 256;
    *blocks = (int)((size + *threads - 1) / *threads);
}

/* Write pattern from GPU */
static int gpu_fill_memory(void *ptr, size_t size, unsigned char value)
{
    int blocks, threads;
    get_launch_config(size, &blocks, &threads);
    
    gpu_write_pattern<<<blocks, threads>>>((unsigned char *)ptr, size, value);
    
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

/* Verify pattern from GPU */
static int gpu_check_memory(void *ptr, size_t size, unsigned char expected)
{
    int *d_result;
    int h_result = 1;  /* Assume success */
    
    /* Allocate result flag in managed memory */
    cudaError_t err = cudaMallocManaged(&d_result, sizeof(int));
    if (err != cudaSuccess) return -1;
    
    *d_result = 1;  /* Initialize to success */
    
    int blocks, threads;
    get_launch_config(size, &blocks, &threads);
    
    gpu_verify_pattern<<<blocks, threads>>>((unsigned char *)ptr, size, 
                                             expected, d_result);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_result);
        return -1;
    }
    
    h_result = *d_result;
    cudaFree(d_result);
    
    return h_result;  /* 1 = success, 0 = mismatch */
}

/* Read-modify-write from GPU */
static int gpu_increment_memory(void *ptr, size_t size)
{
    int blocks, threads;
    get_launch_config(size, &blocks, &threads);
    
    gpu_increment_bytes<<<blocks, threads>>>((unsigned char *)ptr, size);
    
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

/* Verify memory is accessible by writing and reading back */
static int verify_memory_accessible(void *ptr, size_t size)
{
    if (ptr == NULL)
        return 0;

    unsigned char *p = (unsigned char *)ptr;
    
    /* Write pattern */
    for (size_t i = 0; i < size; i++) {
        p[i] = (unsigned char)(i & 0xFF);
    }

    /* Read back and verify */
    for (size_t i = 0; i < size; i++) {
        if (p[i] != (unsigned char)(i & 0xFF))
            return 0;
    }

    return 1;
}

/* Check if CUDA is available */
static int cuda_available(void)
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

/*******************************************************************************
 * Test Cases
 ******************************************************************************/

/* Test: Basic GPU region creation and destruction */
START_TEST(test_gpu_region_create_destroy)
{
    if (!cuda_available()) {
        ck_abort_msg("CUDA not available, skipping GPU tests");
    }

    dstm_gpu_region_t *gr = NULL;
    int err = dstm_region_create_gpu(&gr, 4096);
    ck_assert_int_eq(err, DSTM_SUCCESS);
    ck_assert_ptr_nonnull(gr);
    ck_assert_ptr_nonnull(gr->region);
    ck_assert_ptr_nonnull(gr->gpu_buffer);
    ck_assert_uint_eq(gr->allocated_size, 4096);

    dstm_region_destroy_gpu(&gr);
    ck_assert_ptr_null(gr);
}
END_TEST

/* Test: Simple malloc on GPU region */
START_TEST(test_gpu_region_malloc_simple)
{
    if (!cuda_available()) {
        ck_abort_msg("CUDA not available, skipping GPU tests");
    }

    dstm_gpu_region_t *gr = NULL;
    dstm_region_create_gpu(&gr, 4096);

    dstm_ptr_t ptr = dstm_region_malloc(gr->region, 128);
    ck_assert(!dstm_ptr_isnull(ptr));
    
    void *vptr = dstm_region_ptr_tovoid(gr->region, ptr);
    ck_assert_ptr_nonnull(vptr);
    
    /* Verify we can access this memory from CPU (UVM) */
    ck_assert(verify_memory_accessible(vptr, 128));

    dstm_region_destroy_gpu(&gr);
}
END_TEST

/* Test: Multiple allocations on GPU region */
START_TEST(test_gpu_region_malloc_multiple)
{
    if (!cuda_available()) {
        ck_abort_msg("CUDA not available, skipping GPU tests");
    }

    dstm_gpu_region_t *gr = NULL;
    dstm_region_create_gpu(&gr, 8192);

    dstm_ptr_t ptrs[4];
    void *vptrs[4];
    size_t sizes[] = {64, 128, 256, 512};

    /* Allocate multiple blocks */
    for (int i = 0; i < 4; i++) {
        ptrs[i] = dstm_region_malloc(gr->region, sizes[i]);
        ck_assert(!dstm_ptr_isnull(ptrs[i]));
        
        vptrs[i] = dstm_region_ptr_tovoid(gr->region, ptrs[i]);
        ck_assert_ptr_nonnull(vptrs[i]);
        ck_assert(verify_memory_accessible(vptrs[i], sizes[i]));
    }

    /* Verify allocations don't overlap */
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            char *pi = (char *)vptrs[i];
            char *pj = (char *)vptrs[j];
            /* Check no overlap */
            ck_assert(pi + sizes[i] <= pj || pj + sizes[j] <= pi);
        }
    }

    dstm_region_destroy_gpu(&gr);
}
END_TEST

/* Test: Malloc and free on GPU region */
START_TEST(test_gpu_region_malloc_free)
{
    if (!cuda_available()) {
        ck_abort_msg("CUDA not available, skipping GPU tests");
    }

    dstm_gpu_region_t *gr = NULL;
    dstm_region_create_gpu(&gr, 4096);

    /* Allocate */
    dstm_ptr_t ptr1 = dstm_region_malloc(gr->region, 128);
    ck_assert(!dstm_ptr_isnull(ptr1));

    /* Free */
    dstm_region_free(gr->region, ptr1);

    /* Allocate again - should succeed and possibly reuse memory */
    dstm_ptr_t ptr2 = dstm_region_malloc(gr->region, 128);
    ck_assert(!dstm_ptr_isnull(ptr2));

    void *vptr2 = dstm_region_ptr_tovoid(gr->region, ptr2);
    ck_assert(verify_memory_accessible(vptr2, 128));

    dstm_region_destroy_gpu(&gr);
}
END_TEST

/* Test: GPU sync after CPU writes */
START_TEST(test_gpu_region_sync)
{
    if (!cuda_available()) {
        ck_abort_msg("CUDA not available, skipping GPU tests");
    }

    dstm_gpu_region_t *gr = NULL;
    dstm_region_create_gpu(&gr, 4096);

    dstm_ptr_t ptr = dstm_region_malloc(gr->region, 256);
    ck_assert(!dstm_ptr_isnull(ptr));

    void *vptr = dstm_region_ptr_tovoid(gr->region, ptr);
    
    /* Write data from CPU */
    memset(vptr, 0xAB, 256);

    /* Sync to ensure GPU sees the data */
    int sync_err = dstm_region_gpu_sync();
    ck_assert_int_eq(sync_err, 0);

    /* Verify data persists */
    unsigned char *p = (unsigned char *)vptr;
    for (int i = 0; i < 256; i++) {
        ck_assert_uint_eq(p[i], 0xAB);
    }

    dstm_region_destroy_gpu(&gr);
}
END_TEST

/* Test: Prefetch operations */
START_TEST(test_gpu_region_prefetch)
{
    if (!cuda_available()) {
        ck_abort_msg("CUDA not available, skipping GPU tests");
    }

    dstm_gpu_region_t *gr = NULL;
    dstm_region_create_gpu(&gr, 4096);

    dstm_ptr_t ptr = dstm_region_malloc(gr->region, 512);
    void *vptr = dstm_region_ptr_tovoid(gr->region, ptr);
    
    /* Write some data */
    memset(vptr, 0xCD, 512);

    /* Prefetch to GPU device 0 */
    int err = dstm_region_prefetch_to_gpu(gr, 0);
    ck_assert_int_eq(err, 0);
    
    dstm_region_gpu_sync();

    /* Prefetch back to CPU */
    err = dstm_region_prefetch_to_cpu(gr);
    ck_assert_int_eq(err, 0);
    
    dstm_region_gpu_sync();

    /* Data should still be valid */
    unsigned char *p = (unsigned char *)vptr;
    for (int i = 0; i < 512; i++) {
        ck_assert_uint_eq(p[i], 0xCD);
    }

    dstm_region_destroy_gpu(&gr);
}
END_TEST

/* Test: Large allocation on GPU */
START_TEST(test_gpu_region_large_allocation)
{
    if (!cuda_available()) {
        ck_abort_msg("CUDA not available, skipping GPU tests");
    }

    /* Create 1MB region */
    dstm_gpu_region_t *gr = NULL;
    int err = dstm_region_create_gpu(&gr, 1024 * 1024);
    ck_assert_int_eq(err, DSTM_SUCCESS);

    /* Allocate 512KB */
    dstm_ptr_t ptr = dstm_region_malloc(gr->region, 512 * 1024);
    ck_assert(!dstm_ptr_isnull(ptr));

    void *vptr = dstm_region_ptr_tovoid(gr->region, ptr);
    ck_assert_ptr_nonnull(vptr);

    /* Write pattern to large buffer */
    memset(vptr, 0xEF, 512 * 1024);
    dstm_region_gpu_sync();

    /* Spot check some values */
    unsigned char *p = (unsigned char *)vptr;
    ck_assert_uint_eq(p[0], 0xEF);
    ck_assert_uint_eq(p[1000], 0xEF);
    ck_assert_uint_eq(p[100000], 0xEF);
    ck_assert_uint_eq(p[512 * 1024 - 1], 0xEF);

    dstm_region_destroy_gpu(&gr);
}
END_TEST

/* Test: Allocation exhaustion */
START_TEST(test_gpu_region_exhaustion)
{
    if (!cuda_available()) {
        ck_abort_msg("CUDA not available, skipping GPU tests");
    }

    /* Small region */
    dstm_gpu_region_t *gr = NULL;
    dstm_region_create_gpu(&gr, 256);

    /* Try to allocate more than available */
    dstm_ptr_t ptr = dstm_region_malloc(gr->region, 512);
    ck_assert(dstm_ptr_isnull(ptr));

    /* Smaller allocation should still work */
    ptr = dstm_region_malloc(gr->region, 64);
    ck_assert(!dstm_ptr_isnull(ptr));

    dstm_region_destroy_gpu(&gr);
}
END_TEST

/* Test: Pointer conversion round-trip */
START_TEST(test_gpu_region_ptr_roundtrip)
{
    if (!cuda_available()) {
        ck_abort_msg("CUDA not available, skipping GPU tests");
    }

    dstm_gpu_region_t *gr = NULL;
    dstm_region_create_gpu(&gr, 4096);

    dstm_ptr_t ptr = dstm_region_malloc(gr->region, 128);
    ck_assert(!dstm_ptr_isnull(ptr));

    /* Convert to void* and back */
    void *vptr = dstm_region_ptr_tovoid(gr->region, ptr);
    ck_assert_ptr_nonnull(vptr);

    dstm_ptr_t ptr2 = dstm_region_ptr_fromvoid(gr->region, vptr);
    
    /* Should get same logical pointer back */
    ck_assert_int_eq(ptr.region, ptr2.region);
    ck_assert_int_eq(ptr.offset, ptr2.offset);

    dstm_region_destroy_gpu(&gr);
}
END_TEST

/* Test: Invalid parameters */
START_TEST(test_gpu_region_invalid_params)
{
    if (!cuda_available()) {
        ck_abort_msg("CUDA not available, skipping GPU tests");
    }

    /* NULL output pointer */
    int err = dstm_region_create_gpu(NULL, 4096);
    ck_assert_int_ne(err, DSTM_SUCCESS);

    /* Zero size */
    dstm_gpu_region_t *gr = NULL;
    err = dstm_region_create_gpu(&gr, 0);
    ck_assert_int_ne(err, DSTM_SUCCESS);

    /* Destroy NULL should not crash */
    dstm_region_destroy_gpu(NULL);
    
    dstm_gpu_region_t *null_gr = NULL;
    dstm_region_destroy_gpu(&null_gr);
}
END_TEST

/* Test: CPU writes, GPU reads */
START_TEST(test_gpu_region_cpu_to_gpu)
{
    if (!cuda_available()) {
        ck_abort_msg("CUDA not available");
    }

    dstm_gpu_region_t *gr = NULL;
    dstm_region_create_gpu(&gr, 4096);

    dstm_ptr_t ptr = dstm_region_malloc(gr->region, 1024);
    ck_assert(!dstm_ptr_isnull(ptr));

    void *vptr = dstm_region_ptr_tovoid(gr->region, ptr);
    ck_assert_ptr_nonnull(vptr);

    /* CPU writes pattern */
    memset(vptr, 0xAA, 1024);

    /* GPU verifies the pattern */
    int gpu_result = gpu_check_memory(vptr, 1024, 0xAA);
    ck_assert_int_eq(gpu_result, 1);  /* GPU should see CPU's writes */

    dstm_region_destroy_gpu(&gr);
}
END_TEST

/* Test: GPU writes, CPU reads */
START_TEST(test_gpu_region_gpu_to_cpu)
{
    if (!cuda_available()) {
        ck_abort_msg("CUDA not available");
    }

    dstm_gpu_region_t *gr = NULL;
    dstm_region_create_gpu(&gr, 4096);

    dstm_ptr_t ptr = dstm_region_malloc(gr->region, 1024);
    void *vptr = dstm_region_ptr_tovoid(gr->region, ptr);
    
    /* Initialize to zero from CPU */
    memset(vptr, 0x00, 1024);

    /* GPU writes pattern */
    int err = gpu_fill_memory(vptr, 1024, 0xBB);
    ck_assert_int_eq(err, 0);

    /* CPU verifies the pattern */
    unsigned char *p = (unsigned char *)vptr;
    for (int i = 0; i < 1024; i++) {
        ck_assert_uint_eq(p[i], 0xBB);  /* CPU should see GPU's writes */
    }

    dstm_region_destroy_gpu(&gr);
}
END_TEST

/* Test: GPU read-modify-write */
START_TEST(test_gpu_region_read_modify_write)
{
    if (!cuda_available()) {
        ck_abort_msg("CUDA not available");
    }

    dstm_gpu_region_t *gr = NULL;
    dstm_region_create_gpu(&gr, 4096);

    dstm_ptr_t ptr = dstm_region_malloc(gr->region, 256);
    void *vptr = dstm_region_ptr_tovoid(gr->region, ptr);
    
    /* CPU sets initial values */
    unsigned char *p = (unsigned char *)vptr;
    for (int i = 0; i < 256; i++) {
        p[i] = (unsigned char)i;
    }

    /* GPU increments each byte */
    int err = gpu_increment_memory(vptr, 256);
    ck_assert_int_eq(err, 0);

    /* CPU verifies incremented values */
    for (int i = 0; i < 256; i++) {
        unsigned char expected = (unsigned char)((i + 1) & 0xFF);
        ck_assert_uint_eq(p[i], expected);
    }

    dstm_region_destroy_gpu(&gr);
}
END_TEST

/* Test: Multiple GPU allocations, GPU accesses all */
START_TEST(test_gpu_region_multiple_gpu_access)
{
    if (!cuda_available()) {
        ck_abort_msg("CUDA not available");
    }

    dstm_gpu_region_t *gr = NULL;
    dstm_region_create_gpu(&gr, 16384);

    dstm_ptr_t ptrs[4];
    void *vptrs[4];
    size_t sizes[] = {256, 512, 1024, 2048};
    unsigned char patterns[] = {0x11, 0x22, 0x33, 0x44};

    /* Allocate multiple blocks */
    for (int i = 0; i < 4; i++) {
        ptrs[i] = dstm_region_malloc(gr->region, sizes[i]);
        ck_assert(!dstm_ptr_isnull(ptrs[i]));
        vptrs[i] = dstm_region_ptr_tovoid(gr->region, ptrs[i]);
    }

    /* GPU writes different pattern to each block */
    for (int i = 0; i < 4; i++) {
        int err = gpu_fill_memory(vptrs[i], sizes[i], patterns[i]);
        ck_assert_int_eq(err, 0);
    }

    /* CPU verifies each block has correct pattern */
    for (int i = 0; i < 4; i++) {
        unsigned char *p = (unsigned char *)vptrs[i];
        for (size_t j = 0; j < sizes[i]; j++) {
            ck_assert_uint_eq(p[j], patterns[i]);
        }
    }

    dstm_region_destroy_gpu(&gr);
}
END_TEST

/* Test: Prefetch improves GPU access (functional, not performance) */
START_TEST(test_gpu_region_prefetch_and_access)
{
    if (!cuda_available()) {
        ck_abort_msg("CUDA not available");
    }

    dstm_gpu_region_t *gr = NULL;
    dstm_region_create_gpu(&gr, 4096);

    dstm_ptr_t ptr = dstm_region_malloc(gr->region, 1024);
    void *vptr = dstm_region_ptr_tovoid(gr->region, ptr);

    /* CPU writes data */
    memset(vptr, 0x55, 1024);

    /* Prefetch to GPU */
    int err = dstm_region_prefetch_to_gpu(gr, 0);
    ck_assert_int_eq(err, 0);
    dstm_region_gpu_sync();

    /* GPU verifies data (should be fast since prefetched) */
    int result = gpu_check_memory(vptr, 1024, 0x55);
    ck_assert_int_eq(result, 1);

    /* GPU modifies data */
    err = gpu_fill_memory(vptr, 1024, 0x66);
    ck_assert_int_eq(err, 0);

    /* Prefetch back to CPU */
    err = dstm_region_prefetch_to_cpu(gr);
    ck_assert_int_eq(err, 0);
    dstm_region_gpu_sync();

    /* CPU verifies GPU's modifications */
    unsigned char *p = (unsigned char *)vptr;
    for (int i = 0; i < 1024; i++) {
        ck_assert_uint_eq(p[i], 0x66);
    }

    dstm_region_destroy_gpu(&gr);
}
END_TEST

/*******************************************************************************
 * Test Suite Setup
 ******************************************************************************/

Suite *dstm_gpu_region_suite(void)
{
    Suite *s = suite_create("DSTM GPU Region");

    /* Core test case */
    TCase *tc_core = tcase_create("Core");
    tcase_add_test(tc_core, test_gpu_region_create_destroy);
    tcase_add_test(tc_core, test_gpu_region_malloc_simple);
    tcase_add_test(tc_core, test_gpu_region_malloc_multiple);
    tcase_add_test(tc_core, test_gpu_region_malloc_free);
    tcase_add_test(tc_core, test_gpu_region_ptr_roundtrip);
    tcase_add_test(tc_core, test_gpu_region_invalid_params);
    suite_add_tcase(s, tc_core);

    /* GPU-specific operations */
    TCase *tc_gpu = tcase_create("GPU Operations");
    tcase_add_test(tc_gpu, test_gpu_region_sync);
    tcase_add_test(tc_gpu, test_gpu_region_prefetch);
    tcase_add_test(tc_gpu, test_gpu_region_cpu_to_gpu);
    tcase_add_test(tc_gpu, test_gpu_region_gpu_to_cpu);
    tcase_add_test(tc_gpu, test_gpu_region_read_modify_write);
    tcase_add_test(tc_gpu, test_gpu_region_multiple_gpu_access);
    tcase_add_test(tc_gpu, test_gpu_region_prefetch_and_access);
    suite_add_tcase(s, tc_gpu);

    /* Stress tests */
    TCase *tc_stress = tcase_create("Stress");
    tcase_set_timeout(tc_stress, 30);  /* Longer timeout for large allocations */
    tcase_add_test(tc_stress, test_gpu_region_large_allocation);
    tcase_add_test(tc_stress, test_gpu_region_exhaustion);
    suite_add_tcase(s, tc_stress);

    return s;
}

/*******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char **argv)
{
    /* Check CUDA availability early */
    if (!cuda_available()) {
        fprintf(stderr, "WARNING: No CUDA devices available.\n");
        fprintf(stderr, "GPU tests will be skipped.\n");
    } else {
        int device;
        cudaGetDevice(&device);
        
        struct cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);
        
        printf("Running GPU tests on: %s\n", props.name);
        printf("  Compute capability: %d.%d\n", props.major, props.minor);
        printf("  Total memory: %.2f GB\n", 
               props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Managed memory: %s\n", 
               props.managedMemory ? "supported" : "not supported");
        printf("\n");
    }

    /* Initialize DSTM global context if needed */
    /* dstm_init(...); */

    Suite *s = dstm_gpu_region_suite();
    SRunner *sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    int number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    /* Cleanup DSTM if needed */
    /* dstm_finalize(); */

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
