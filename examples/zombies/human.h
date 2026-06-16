#ifndef ZOMBIE_HUMAN_H
#define ZOMBIE_HUMAN_H

#include "event.h"

/* Humans
 * - A human has a home cell (home_node.home_local).
 * - Each day they attend a morning event and an afternoon event,
 *   returning home in between and at night.
 * - A human becomes infected if they share a (node, local) cell with
 *   a zombie (or contagious infected human) at the same hour.
 * - Infected humans become zombies after INFECT_DAYS_TO_ZOMBIE days.
 * - Zombies stop attending events and just shamble in place.
 *
 * Like flight_t in the airport sim, a human carries its full state
 * (state, destination, legs) inside the bundle, so any rank along
 * the route can forward it without consulting the owner.
 */

#define INFECT_DAYS_TO_ZOMBIE   3
#define PAYLOAD_MAX             64

typedef enum {
    HSTATE_HEALTHY  = 0,
    HSTATE_INFECTED = 1,   /* contagious, still walking, ticking down */
    HSTATE_ZOMBIE   = 2
} human_state_t;

typedef struct human_t {
    int           valid;            /* slot occupied? */
    int           human_id;         /* globally unique */
    human_state_t state;
    int           days_infected;    /* 0 until infected */
    int           last_infect_day;  /* day on which days_infected last advanced */

    /* Where the human currently *is* (cell they occupy this hour). */
    int           cur_node;
    int           cur_local;

    /* Where the human is trying to go this hop (set by the owner). */
    int           dest_node;
    int           dest_local;

    /* Routing: next hop rank chosen by the mesh-routing function.
     * Mirrors flight_t.next_hop_rank. */
    int           next_hop_rank;
    int           legs_completed;

    /* Home + today's schedule. The owner re-rolls each new day. */
    int           home_node;
    int           home_local;
    event_t       morning_event;
    event_t       afternoon_event;

    /* Bookkeeping for logging, like flight_t.payload. */
    char          payload[PAYLOAD_MAX];
} human_t;

#endif /* ZOMBIE_HUMAN_H */
