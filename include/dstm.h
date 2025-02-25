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
//
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

#define dstm_log_error(...)                                                     \
	dstm_log_printf(DSTM_LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define dstm_log_warning(...)                                                   \
	dstm_log_printf(DSTM_LOG_WARNING, __FILE__, __LINE__, __VA_ARGS__)
#define dstm_log_normal(...)                                                    \
	dstm_log_printf(DSTM_LOG_NORMAL, __FILE__, __LINE__, __VA_ARGS__)
#define dstm_log_info(...)                                                      \
	dstm_log_printf(DSTM_LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define dstm_log_debug(...)                                                     \
	dstm_log_printf(DSTM_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define dstm_log_perror(...)                                                    \
	do {                                                                   \
		char *__dstm_errstr = strerror(errno);                          \
		dstm_log_printf(DSTM_LOG_ERROR, __FILE__, __LINE__,              \
		               __VA_ARGS__);                                   \
		dstm_log_printf(DSTM_LOG_ERROR, __FILE__, __LINE__,              \
		               "perror: %s\n", __dstm_errstr);                  \
	} while (0)


#ifdef __cplusplus
}
#endif

#endif
