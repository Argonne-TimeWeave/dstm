#ifndef INC_airport_h
#define INC_airport_h

#include <ross.h>

/* --- Tunables (also settable via command line) --- */
#define TIER_HUB       0
#define TIER_REGIONAL  1
#define TIER_LOCAL     2

/* Percent chance a HUB/REGIONAL/LOCAL airport initiates a departure per "tick" */
static const int tier_send_prob[3]   = { 80, 40, 15 };
/* Weight used when picking a destination airport by tier */
static const int tier_recv_weight[3] = {  8,  3,  1 };

static const char * const tier_name[3] = { "HUB", "REGIONAL", "LOCAL" };

/* Event types */
typedef enum {
    EV_TICK,      /* self-scheduled: "consider departing" */
    EV_ARRIVAL,   /* a plane arrived at the final dest airport */
    EV_CONNECT    /* a plane is passing through as a layover */
} airport_evtype_t;

/* Per-LP (per-airport) state */
typedef struct {
    int tier;              /* HUB/REGIONAL/LOCAL, decided in init from gid + seed */
    unsigned long departures;
    unsigned long arrivals;
    unsigned long connections;
} airport_state;

/* Event payload */
typedef struct {
    airport_evtype_t type;
    tw_lpid  origin;       /* original source LP */
    tw_lpid  final_dest;   /* final destination LP */
    int      legs;         /* number of hops completed so far */

    /* Reverse-computation bookkeeping */
    int      rc_did_depart;  /* whether the TICK actually caused a departure */
    unsigned rc_rng_calls;   /* how many tw_rand_* were called; for reverse */
} airport_message;

/* Globals (configurable via tw_optdef) */
extern unsigned int nairports;          /* total LPs = mesh_rows * mesh_cols */
extern unsigned int mesh_rows;
extern unsigned int mesh_cols;
extern unsigned int airports_per_rank;
extern unsigned int layover_threshold;  /* Manhattan dist > this ⇒ unreachable */
extern unsigned int work_iters;         /* synthetic FP work per departure */
extern double       mean_interdeparture;/* mean of exp() between TICKs */
extern double       mean_flight_time;   /* mean of exp() per hop */

#endif
