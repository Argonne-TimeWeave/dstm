#ifndef ZOMBIE_HUMAN_H
#define ZOMBIE_HUMAN_H

/* Humans
 * - A human occupies one (node, local) cell.
 * - Each tick a human moves toward a NEIGHBORING local cell with the
 *   fewest zombies (Moore 8-neighborhood). A zombie moves toward the
 *   neighboring cell with the most humans. Humans may take up to
 *   HUMAN_STEP local hops per tick; zombies take ZOMBIE_STEP.
 * - A zombie sharing a cell with >=1 healthy humans infects exactly
 *   ONE of them per tick (Repast semantics: humans.oneOf()).
 * - Infected humans become zombies after INFECT_TICKS_TO_ZOMBIE ticks.
 *
 * Like flight_t in the airport sim, a human carries its full state
 * inside the bundle so any rank along the route can forward it
 * without consulting the owner.
 */

#define INFECT_TICKS_TO_ZOMBIE  50
#define PAYLOAD_MAX             64

/* Repast: zombies move .5 per tick, humans up to 1.5.
 * On an integer lattice we approximate that as: zombies take 1 hop,
 * humans take up to 2 hops per tick. Change these to (1,1) to force
 * strict symmetry. */
#define ZOMBIE_STEP  1
#define HUMAN_STEP   2

typedef enum {
    HSTATE_HEALTHY  = 0,
    HSTATE_INFECTED = 1,
    HSTATE_ZOMBIE   = 2
} human_state_t;

typedef struct human_t {
    int           valid;
    int           human_id;
    human_state_t state;
    int           infection_time;    /* ticks since infection */

    /* Where the human currently is (cell they occupy this tick). */
    int           cur_node;
    int           cur_local;

    /* Where the human is trying to go this hop. */
    int           dest_node;
    int           dest_local;

    /* Routing: next hop rank chosen by the mesh-routing function. */
    int           next_hop_rank;
    int           legs_completed;

    /* Home cell (starting position). */
    int           home_node;
    int           home_local;

    /* Bookkeeping for logging. */
    char          payload[PAYLOAD_MAX];
} human_t;

#endif /* ZOMBIE_HUMAN_H */
