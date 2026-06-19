/*
 * Minimal GPU-aware ring network simulation with DSTM.
 *
 * Differences from mini.c:
 *   - Each rank owns a dstm_gpu_region_t (CUDA Unified Memory) used as a
 *     payload arena. Packet payloads are allocated from this arena so a
 *     GPU kernel can read them directly.
 *   - Mailbox entries no longer carry inline payload bytes. Instead they
 *     carry a (region-local) dstm_ptr_t handle plus a payload_len. The
 *     receiving rank converts the handle to a void* via its OWN region.
 *   - When a packet is DELIVERED (dest == rank), we launch a GPU kernel
 *     that computes a checksum over the payload. The CPU then recomputes
 *     the same checksum and asserts equality.
 *   - When a packet is FORWARDED, we allocate a fresh payload buffer in
 *     the local arena and copy the bytes in (UVM makes this a plain
 *     memcpy). The old buffer is freed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <mpi.h>
#include <cuda_runtime.h>

extern "C" {
#include "dstm.h"
#include "region.h"
}

#define PACKET_PAYLOAD  240   /* bytes per packet payload                 */
#define SLOT_CAPACITY   8     /* max packets per slot per timestep        */
#define ARENA_BYTES     (1u << 20) /* 1 MiB UVM arena per rank            */

#define MAILBOX_SLOTS   2
#define SLOT_LEFT       0
#define SLOT_RIGHT      1

/* ---------------- forward decls of the GPU helpers from your test file -- */
typedef struct {
    dstm_region_t *region;
    void          *gpu_buffer;
    size_t         allocated_size;
} dstm_gpu_region_t;

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

/* ============================================================ GPU KERNEL */
/*
 * Compute a 32-bit additive checksum of `len` bytes at `payload`.
 * Two-stage: each block does a partial sum in shared memory, then atomically
 * adds into a single global accumulator. Result is written to *out.
 */
__global__ void gpu_checksum_kernel(const unsigned char *payload,
                                    int len,
                                    unsigned int *out)
{
    __shared__ unsigned int part;
    if (threadIdx.x == 0) part = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) atomicAdd(&part, (unsigned int)payload[idx]);
    __syncthreads();

    if (threadIdx.x == 0) atomicAdd(out, part);
}

static unsigned int gpu_checksum(const unsigned char *uvm_ptr, int len)
{
    unsigned int *d_out;
    cudaMallocManaged(&d_out, sizeof(unsigned int));
    *d_out = 0;

    int threads = 256;
    int blocks  = (len + threads - 1) / threads;
    gpu_checksum_kernel<<<blocks, threads>>>(uvm_ptr, len, d_out);
    cudaDeviceSynchronize();

    unsigned int v = *d_out;
    cudaFree(d_out);
    return v;
}

static unsigned int cpu_checksum(const unsigned char *p, int len)
{
    unsigned int s = 0;
    for (int i = 0; i < len; i++) s += p[i];
    return s;
}

/* ============================================================ MAILBOX  */
/*
 * A packet now carries a HANDLE (dstm_ptr_t) into the receiving rank's
 * arena. When forwarding, the receiver allocates a new buffer in its OWN
 * arena and copies the bytes; the producing rank fills in the handle
 * with respect to *its own* arena before PUTting.
 *
 * To keep the wire layout self-contained we also embed the payload bytes
 * directly in the slot. The recipient copies them into its arena on
 * arrival, then runs the GPU kernel on the arena copy. This keeps the
 * DSTM PUT/GET semantics simple (slots are POD blobs) while still
 * exercising GPU memory + a GPU kernel per arrival.
 */
typedef struct {
    int            valid;
    int            source_rank;
    int            dest_rank;
    int            hops_taken;
    int            payload_len;
    unsigned int   cpu_checksum;        /* computed by sender once       */
    dstm_time_t    origin_time;
    unsigned char  payload[PACKET_PAYLOAD];
} mailbox_entry_t;

typedef struct {
    int             n_packets;
    mailbox_entry_t packets[SLOT_CAPACITY];
} mailbox_slot_t;

static mailbox_slot_t empty_slot = { 0 };

/* ============================================================ MAIN     */
static void usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s [--seed S] [--time T] [--prob P] [--quiet]\n", prog);
}

int main(int argc, char *argv[])
{
    int seed = 42, quiet = 0, timesteps = 16, send_prob = 60;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--time") && i+1<argc) timesteps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prob") && i+1<argc) send_prob = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]); return 0;
        }
    }

    int rank, num_ranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    /* Pick a CUDA device (one per rank, round-robin) */
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    if (ndev > 0) cudaSetDevice(rank % ndev);

    dstm_init(&argc, &argv);
    dstm_log_init(stdout, "network_sim_gpu");
    dstm_log_setlevel(quiet ? DSTM_LOG_QUIET : DSTM_LOG_NORMAL);

    /* ---- topology + start ---------------------------------------- */
    dstm_distance_t *topology;
    if (dstm_distance_create_ring(&topology, num_ranks) != DSTM_SUCCESS) {
        dstm_log_error("ring create failed"); MPI_Finalize(); return -1;
    }
    if (rank == 0)
        dstm_log_normal("Ring + GPU: %d nodes, %d packets/slot, "
                        "%u-byte payload, %u-byte arena\n",
                        num_ranks, SLOT_CAPACITY,
                        (unsigned)PACKET_PAYLOAD, (unsigned)ARENA_BYTES);

    if (dstm_start(topology) != DSTM_SUCCESS) {
        dstm_log_error("dstm_start failed"); return -1;
    }

    /* ---- GPU arena (UVM) for this rank --------------------------- */
    dstm_gpu_region_t *arena = NULL;
    if (dstm_region_create_gpu(&arena, ARENA_BYTES) != DSTM_SUCCESS) {
        dstm_log_error("GPU arena create failed"); return -1;
    }
    dstm_region_prefetch_to_gpu(arena, rank % (ndev ? ndev : 1));

    /* ---- distributed mailbox array ------------------------------- */
    dstm_darray_t *mailbox;
    if (dstm_darray_create_equal(&mailbox,
                                 sizeof(mailbox_slot_t),
                                 (size_t)num_ranks * MAILBOX_SLOTS)
        != DSTM_SUCCESS)
    {
        dstm_log_error("mailbox create failed"); return -1;
    }

    srand(seed + rank);
    int left_rank  = (rank - 1 + num_ranks) % num_ranks;
    int right_rank = (rank + 1) % num_ranks;

    mailbox_slot_t outbox[MAILBOX_SLOTS];
    mailbox_slot_t recv_slot;

    long delivered = 0, mismatched = 0;

    for (dstm_time_t t = 0; t < timesteps; t++) {
        dstm_log_normal("Rank %d: step %ld\n", rank, t);

        /* STEP 1: sentinel PUT --------------------------------------- */
        for (int s = 0; s < MAILBOX_SLOTS; s++) {
            size_t idx = (size_t)rank * MAILBOX_SLOTS + s;
            dstm_darray_put(mailbox, idx, &empty_slot, t, 0);
        }

        /* STEP 2: receive both slots --------------------------------- */
        outbox[SLOT_LEFT]  = empty_slot;
        outbox[SLOT_RIGHT] = empty_slot;

        for (int s = 0; s < MAILBOX_SLOTS; s++) {
            size_t idx = (size_t)rank * MAILBOX_SLOTS + s;
            int st = dstm_darray_get_or_default(mailbox, idx,
                                                &recv_slot, t, 0,
                                                &empty_slot);
            if (st != DSTM_SUCCESS) {
                dstm_log_error("GET failed slot %d", s); continue;
            }

            for (int p = 0; p < recv_slot.n_packets; p++) {
                mailbox_entry_t pkt = recv_slot.packets[p];

                if (pkt.dest_rank == rank) {
                    /* ---- DELIVER: stage payload into GPU arena and
                     *      run the checksum kernel; validate on CPU. */
                    dstm_ptr_t gptr =
                        dstm_region_malloc(arena->region, pkt.payload_len);
                    if (dstm_ptr_isnull(gptr)) {
                        dstm_log_warning("arena full on deliver");
                        continue;
                    }
                    unsigned char *gbuf = (unsigned char *)
                        dstm_region_ptr_tovoid(arena->region, gptr);
                    memcpy(gbuf, pkt.payload, pkt.payload_len);

                    unsigned int gsum = gpu_checksum(gbuf, pkt.payload_len);
                    unsigned int csum = cpu_checksum(pkt.payload,
                                                     pkt.payload_len);

                    if (gsum == csum && csum == pkt.cpu_checksum) {
                        dstm_log_normal(
                          "Rank %d [t=%ld]: DELIVERED src=%d hops=%d "
                          "len=%d cksum=0x%08x OK\n",
                          rank, t, pkt.source_rank, pkt.hops_taken,
                          pkt.payload_len, gsum);
                        delivered++;
                    } else {
                        dstm_log_error(
                          "Rank %d [t=%ld]: CHECKSUM MISMATCH "
                          "src=%d gpu=0x%08x cpu=0x%08x sent=0x%08x\n",
                          rank, t, pkt.source_rank,
                          gsum, csum, pkt.cpu_checksum);
                        mismatched++;
                    }

                    dstm_region_free(arena->region, gptr);
                } else {
                    /* ---- FORWARD one hop --------------------------- */
                    int dist_right = (pkt.dest_rank - rank + num_ranks)
                                     % num_ranks;
                    int go_right   = (dist_right <= num_ranks - dist_right);
                    int fwd_dir    = go_right ? SLOT_RIGHT : SLOT_LEFT;

                    pkt.hops_taken++;
                    mailbox_slot_t *ob = &outbox[fwd_dir];
                    if (ob->n_packets < SLOT_CAPACITY) {
                        ob->packets[ob->n_packets++] = pkt;
                        dstm_log_normal(
                          "Rank %d [t=%ld]: FORWARD src=%d dst=%d via %s "
                          "(hop %d)\n",
                          rank, t, pkt.source_rank, pkt.dest_rank,
                          go_right ? "right" : "left", pkt.hops_taken);
                    } else {
                        dstm_log_warning(
                          "Rank %d [t=%ld]: DROP src=%d dst=%d outbox full\n",
                          rank, t, pkt.source_rank, pkt.dest_rank);
                    }
                }
            }
        }

        /* STEP 3: maybe inject -------------------------------------- */
        if (num_ranks > 1 && (rand() % 100) < send_prob) {
            int dest = rand() % (num_ranks - 1);
            if (dest >= rank) dest++;

            int dist_right = (dest - rank + num_ranks) % num_ranks;
            int go_right   = (dist_right <= num_ranks - dist_right);
            int fwd_dir    = go_right ? SLOT_RIGHT : SLOT_LEFT;

            mailbox_slot_t *ob = &outbox[fwd_dir];
            if (ob->n_packets < SLOT_CAPACITY) {
                mailbox_entry_t pkt;
                memset(&pkt, 0, sizeof(pkt));
                pkt.valid       = 1;
                pkt.source_rank = rank;
                pkt.dest_rank   = dest;
                pkt.hops_taken  = 0;
                pkt.origin_time = t;

                /* Build a payload — fill with a deterministic-ish pattern
                 * so checksum is meaningful. */
                int len = 32 + (rand() % (PACKET_PAYLOAD - 32));
                pkt.payload_len = len;
                for (int i = 0; i < len; i++)
                    pkt.payload[i] = (unsigned char)((rank * 17 + i) & 0xFF);

                pkt.cpu_checksum = cpu_checksum(pkt.payload, len);

                ob->packets[ob->n_packets++] = pkt;
                dstm_log_normal(
                  "Rank %d [t=%ld]: INJECT dst=%d len=%d cksum=0x%08x "
                  "via %s\n",
                  rank, t, dest, len, pkt.cpu_checksum,
                  go_right ? "right" : "left");
            }
        }

        /* STEP 4: send outboxes ------------------------------------- */
        if (outbox[SLOT_RIGHT].n_packets > 0) {
            size_t idx = (size_t)right_rank * MAILBOX_SLOTS + SLOT_LEFT;
            if (dstm_darray_put(mailbox, idx,
                                &outbox[SLOT_RIGHT], t, 1) != DSTM_SUCCESS)
                dstm_log_error("PUT right→left failed");
        }
        if (outbox[SLOT_LEFT].n_packets > 0) {
            size_t idx = (size_t)left_rank * MAILBOX_SLOTS + SLOT_RIGHT;
            if (dstm_darray_put(mailbox, idx,
                                &outbox[SLOT_LEFT], t, 1) != DSTM_SUCCESS)
                dstm_log_error("PUT left→right failed");
        }
    }

    /* ---- teardown ------------------------------------------------ */
    dstm_time_t last = timesteps + 1;
    dstm_barrier(last);

    dstm_log_normal("Rank %d: delivered=%ld mismatched=%ld\n",
                    rank, delivered, mismatched);

    dstm_darray_destroy(&mailbox);
    dstm_region_destroy_gpu(&arena);
    dstm_fini(last);
    dstm_distance_destroy(&topology);
    dstm_finalize();
    MPI_Finalize();
    return mismatched == 0 ? 0 : 1;
}
