#ifndef ZOMBIE_EVENT_H
#define ZOMBIE_EVENT_H

/* EVENTS
 * - Humans have two events per day (morning + afternoon).
 * - Time is tracked in whole hours (military time).
 * - An event has a location = <node>.<local>.
 * - Travel cost from human's current cell to an event cell:
 *      dstm_distance_get(cur_node, evt_node) hours between nodes,
 *      + 1 hour to walk from the node edge to the local cell.
 *
 * Morning  events: 3  <= start, end <= 12
 * Afternoon events: 14 <= start, end <= 22
 * Events are <= 3 hours long.
 */

#define HOURS_PER_DAY        24

#define MORNING_START_MIN    3
#define MORNING_END_MAX      12
#define AFTERNOON_START_MIN  14
#define AFTERNOON_END_MAX    22
#define EVENT_MAX_HOURS      3

typedef enum {
    SLOT_MORNING   = 0,
    SLOT_AFTERNOON = 1,
    SLOT_HOME      = 2   /* "event" of being at home overnight */
} event_slot_t;

typedef struct event_t {
    int time_start;       /* hour-of-day, military */
    int time_end;         /* hour-of-day, military, <= start + EVENT_MAX_HOURS */
    int location_node;    /* which rank's local mesh */
    int location_local;   /* which cell in that local mesh */
} event_t;

/* Helpers */
static inline int hour_of_day(int t)    { return t % HOURS_PER_DAY; }
static inline int day_of(int t)         { return t / HOURS_PER_DAY; }

#endif /* ZOMBIE_EVENT_H */
