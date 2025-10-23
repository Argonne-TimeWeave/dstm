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
 *
 * 3 core concepts
 *   - logical regions
 *   - logical processes
 *   - timestamped API
 ******************************************************************************/

struct dstm_communicator;
typedef struct dstm_communicator dstm_comm_t;

extern dstm_comm_t DSTM_COMM_WORLD;

int dstm_comm_rank(dstm_comm_t *comm, int *rank);
int dstm_comm_size(dstm_comm_t *comm, int *size);

int dstm_comm_tree_split(dstm_comm_t *comm, int parent, dstm_comm_t **tree);
int dstm_comm_destroy(dstm_comm_t **comm);

struct dstm_logical_region;
typedef struct dstm_logical_region dstm_lr_t;

struct dstm_physical_location;
typedef struct dstm_physical_location dstm_phy_t;

int dstm_lr_create(dstm_lr_t **lr, dstm_phy_t, size_t size);
void dstm_lr_destroy(dstm_lr_t **lr);

struct dstm_logical_process;
typedef struct dstm_logical_process dstm_lp_t;
typedef int (*dstm_lp_func_t)(dstm_lp_t *self, void *arg);

int dstm_lp_spawn(dstm_lp_t **lp, dstm_lp_func_t func, void *arg);
int dstm_lp_join(dstm_lp_t **lp);

#ifdef __cplusplus
}
#endif

#endif
