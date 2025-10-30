/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

/**
 * Main DSTM header, contains most of the public API
 **/

#ifndef DSTM_H
#define DSTM_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// clang-format off
#include "dstm/error.h"
#include "dstm/version.h"
// clang-format on

/*******************************************************************************
 * Library Init/Fini
 ******************************************************************************/

/** initializes the library, in particular using environment variables to figure
 * out if the library should emit messages at all and if a ratelimit is in
 * place.
 *
 * @param[in,out] argc: a pointer to the number of arguments in *argv
 * @param[in,out] argv: an array of command line options.
 *
 * If the library detects command line options it understands, it will consume
 * those arguments and modify the two variables accordingly. Other arguments
 * will be ignored. It is safe to give a program's main arguments to this
 * library.
 *
 * @return 0 if successful; an error code otherwise.
 **/
int dstm_init(int *argc, char **argv[]);

/**
 * Terminates the library. Do this before an instrumented program exits.
 * @return 0 if successful; an error code otherwise.
 **/
int dstm_finalize(void);

/*******************************************************************************
 * Logging
 ******************************************************************************/

#define DSTM_LOG_QUIET 0
#define DSTM_LOG_ERROR 1
#define DSTM_LOG_WARNING 2
#define DSTM_LOG_NORMAL 3
#define DSTM_LOG_INFO 4
#define DSTM_LOG_DEBUG 5

/**
 * Set to one by the library if at least one dstm_log_init has
 * been called
 */
extern int dstm_log_initialized;

/**
 * Initializes dstm logging
 *
 * @param f: file descriptor
 * @param nm: logging source namespace
 * @return 0 if successful, an error code otherwise
 */
int dstm_log_init(FILE *f, const char *nm);

/**
 * Prints an dstm log message at a log level, labeled with source file and line
 * number.
 *
 * @param level: log level constant
 * @param file: source file label typically ``__FILE__``
 * @param line: source line number label. typically ``__LINE__``
 * @param format: printf formatted string
 */
void dstm_log_printf(int level,
                     const char *file,
                     unsigned int line,
                     const char *format,
                     ...);

int dstm_log_setlevel(int level);

#define dstm_log_error(...)                                                    \
	dstm_log_printf(DSTM_LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define dstm_log_warning(...)                                                  \
	dstm_log_printf(DSTM_LOG_WARNING, __FILE__, __LINE__, __VA_ARGS__)
#define dstm_log_normal(...)                                                   \
	dstm_log_printf(DSTM_LOG_NORMAL, __FILE__, __LINE__, __VA_ARGS__)
#define dstm_log_info(...)                                                     \
	dstm_log_printf(DSTM_LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define dstm_log_debug(...)                                                    \
	dstm_log_printf(DSTM_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define dstm_log_perror(...)                                                   \
	do {                                                                   \
		char *__dstm_errstr = strerror(errno);                         \
		dstm_log_printf(DSTM_LOG_ERROR, __FILE__, __LINE__,            \
		                __VA_ARGS__);                                  \
		dstm_log_printf(DSTM_LOG_ERROR, __FILE__, __LINE__,            \
		                "perror: %s\n", __dstm_errstr);                \
	} while (0)

/*******************************************************************************
 * Basic API
 ******************************************************************************/

/* distance between ranks, expressed as a 1D array of length size *size; */
struct dstm_distance_s {
	int *distances;
	size_t size;
};
typedef struct dstm_distance_s dstm_distance_t;

extern dstm_distance_t DSTM_DISTANCE_IDENTITY;

int dstm_distance_deepcopy(dstm_distance_t dst, dstm_distance_t src);

/* basic configuration of dstm:
 */
int dstm_start(dstm_distance_t dist);
int dstm_fini();

/* a memory region: core object that get updated by the simulation using the
 * DSTM model.
 *
 * Has a unique ID within the DSTM, and a current time. Internally maintains a
 * list of pending requests to R/W to it.
 */

struct dstm_region_s;
typedef struct dstm_region_s dstm_region_t;

/* type alias to a timestamp, this is logical time (within the simulation), not
 * real time. */
typedef int64_t dstm_time_t;

/* allocate space in a region for data */
intptr_t dstm_region_malloc(dstm_region_t *r, size_t size);
void dstm_region_free(dstm_region_t *r, intptr_t ptr);


int dstm_get_local_region_id(void);
dstm_region_t *dstm_get_local_region(void);

int dstm_put(int rank, int region_id, dstm_time_t time, dstm_time_t delta,
	     intptr_t dst, void *src, size_t size);
int dstm_get(int rank, int region_id, dstm_time_t time, dstm_time_t delta,
	     void *dst, intptr_t src, size_t size);

/* still not sure on this one */
int dstm_step(dstm_region_t *r, int delta);

#ifdef __cplusplus
}
#endif

#endif
