/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef DSTM_ERROR_H
#define DSTM_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup dstm_error "DSTM Error Management"
 * @brief DSTM Error Codes and Functions
 *
 * Error codes and error helper functions.
 * As is quite common in C code, error code values are defined in positive,
 * but are returned in negative.
 * @{
 **/

/**
 * Variable set by dstm function calls. dstm_errno should be checked after an dstm
 * function call returning an error and prior to any other dstm call that may
 * overwrite it.
 **/
extern int dstm_errno;

/**
 * Get a string description of an dstm error.
 * @param err: the dstm error number.
 * Returns a static string describing the error.
 **/
const char *dstm_strerror(const int err);

/**
 * Print error on standard error output.
 * "msg": A message to prepend to error message.
 **/
void dstm_perror(const char *msg);

/** Generic value for success **/
#define DSTM_SUCCESS 0

/**
 * Generic value for failure
 * Usually when this is the returned value,
 * the function will detail another way to
 * inspect error.
 **/
#define DSTM_FAILURE 1

/**
 * Not enough memory was available
 * for fulfilling DSTM function call.
 **/
#define DSTM_ENOMEM 2

/**
 * One of the argument provided to
 * DSTM function call was invalid.
 **/
#define DSTM_EINVAL 3

/**
 * One of the arguments provided
 * to DSTM function call has out of bound
 * value.
 **/
#define DSTM_EDOM 4

/**
 * Invoked DSTM abstraction function is actually
 * not implemented for this particular version of
 * DSTM abstraction.
 **/
#define DSTM_ENOTSUP 5

/**
 * Invoked DSTM abstraction function has failed
 * because the resource it works on was busy.
 **/
#define DSTM_EBUSY 6

/**
 * Invoked DSTM abstraction function has failed
 * because the user does not have required permissions.
 **/
#define DSTM_EPERM 7

/**
 * Value not found in the collection
 **/
#define DSTM_ENOTFOUND 8

/**
 * The resource (e.g. a graph edge) already exists at the observed
 * version and cannot be created again.  Returned by insertion
 * operations such as dstm_dgraph_edge_add when the target already
 * exists at the effective time.
 **/
#define DSTM_EEXIST 9

#define DSTM_ENOTLOCAL 10

/**
 * Max allowed value for errors.
 **/
#define DSTM_ERROR_MAX 11

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // DSTM_ERROR_H
