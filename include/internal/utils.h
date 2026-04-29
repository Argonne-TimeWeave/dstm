/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef INTERNAL_UTILS_H
#define INTERNAL_UTILS_H 1

/**
 * DSTM_SNPRINTF_ADVANCE - advance buffer/size after a snprintf-style write.
 *
 * Tracks total bytes written (including overflow) and keeps @buf / @n
 * consistent for subsequent snprintf calls.
 *
 * @buf:     char * buffer pointer   (modified in place)
 * @n:       size_t remaining size   (modified in place)
 * @total:   int    cumulative total (modified in place)
 * @written: int    return value of the latest snprintf call
 */
#define DSTM_SNPRINTF_ADVANCE(buf, n, total, written)                          \
	do {                                                                   \
		int _w = (written);                                            \
		(buf) += (_w < (int)(n) ? _w : (int)(n) - 1);                 \
		(n) = (_w < (int)(n) ? (n) - (size_t)_w : 1);                 \
		(total) += _w;                                                 \
	} while (0)

#endif /* INTERNAL_UTILS_H */
