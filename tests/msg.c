/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <check.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

#include "internal/msg.h"

/*******************************************************************************
 * dstm_msg_fill_hdr tests
 ******************************************************************************/

/* Each valid message type is stored correctly in the header. */
START_TEST(test_fill_hdr_put)
{
	dstm_msg_t m;
	memset(&m, 0, sizeof(m));
	int rc = dstm_msg_fill_hdr(&m, DSTM_MSG_TYPE_PUT, 10, 2, 0, 1);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_int_eq(m.header.type, DSTM_MSG_TYPE_PUT);
	ck_assert_int_eq(m.header.dst_rank, 0);
	ck_assert_int_eq(m.header.src_rank, 1);
	ck_assert_int_eq(m.header.time, 10);
	ck_assert_int_eq(m.header.delta, 2);
}
END_TEST

START_TEST(test_fill_hdr_getreq)
{
	dstm_msg_t m;
	memset(&m, 0, sizeof(m));
	int rc = dstm_msg_fill_hdr(&m, DSTM_MSG_TYPE_GET_REQ, 42, 1, 3, 7);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_int_eq(m.header.type, DSTM_MSG_TYPE_GET_REQ);
	ck_assert_int_eq(m.header.dst_rank, 3);
	ck_assert_int_eq(m.header.src_rank, 7);
	ck_assert_int_eq(m.header.time, 42);
	ck_assert_int_eq(m.header.delta, 1);
}
END_TEST

START_TEST(test_fill_hdr_getrep)
{
	dstm_msg_t m;
	memset(&m, 0, sizeof(m));
	int rc = dstm_msg_fill_hdr(&m, DSTM_MSG_TYPE_GET_REP, 99, 5, 2, 0);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_int_eq(m.header.type, DSTM_MSG_TYPE_GET_REP);
	ck_assert_int_eq(m.header.dst_rank, 2);
	ck_assert_int_eq(m.header.src_rank, 0);
	ck_assert_int_eq(m.header.time, 99);
	ck_assert_int_eq(m.header.delta, 5);
}
END_TEST

/* type == DSTM_MSG_TYPE_MAX is out-of-range and must be rejected. */
START_TEST(test_fill_hdr_invalid_type_max)
{
	dstm_msg_t m;
	memset(&m, 0, sizeof(m));
	int rc = dstm_msg_fill_hdr(&m, DSTM_MSG_TYPE_MAX, 0, 0, 0, 0);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_int_eq(dstm_errno, DSTM_EINVAL);
}
END_TEST

/* Negative type values must also be rejected. */
START_TEST(test_fill_hdr_invalid_type_negative)
{
	dstm_msg_t m;
	memset(&m, 0, sizeof(m));
	int rc = dstm_msg_fill_hdr(&m, -1, 0, 0, 0, 0);
	ck_assert_int_eq(rc, -DSTM_EINVAL);
	ck_assert_int_eq(dstm_errno, DSTM_EINVAL);
}
END_TEST

/*******************************************************************************
 * dstm_msg_fill_payload tests
 *
 * fill_payload is the single fill function for PUT, GET_REQ and GET_REP.
 * It stores the putget body (ptr + size) and sets msg->data to the provided
 * src pointer without copying; the request's buf field is the owner.
 ******************************************************************************/

/* PUT case: ptr and src both set; data must alias src exactly. */
START_TEST(test_fill_payload_put)
{
	dstm_msg_t m;
	memset(&m, 0, sizeof(m));

	char buf[64];
	memset(buf, 0xBB, sizeof(buf));

	dstm_ptr_t dst = {.region = 3, .offset = 128};
	int rc = dstm_msg_fill_payload(&m, dst, buf, sizeof(buf));
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_int_eq(m.u.putget.ptr.region, 3);
	ck_assert_int_eq(m.u.putget.ptr.offset, 128);
	ck_assert_uint_eq(m.u.putget.size, sizeof(buf));
	/* data is a non-owning alias into the caller-supplied buffer */
	ck_assert_ptr_eq(m.data, buf);
}
END_TEST

/* GET_REQ case: src == NULL (no payload); data must be NULL. */
START_TEST(test_fill_payload_getreq)
{
	dstm_msg_t m;
	memset(&m, 0xFF, sizeof(m)); /* start from garbage to confirm NULL */

	dstm_ptr_t src = {.region = 5, .offset = 256};
	int rc = dstm_msg_fill_payload(&m, src, NULL, 32);
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_int_eq(m.u.putget.ptr.region, 5);
	ck_assert_int_eq(m.u.putget.ptr.offset, 256);
	ck_assert_uint_eq(m.u.putget.size, 32);
	/* no payload for a GET request */
	ck_assert_ptr_null(m.data);
}
END_TEST

/* GET_REP case: ptr is not meaningful (pass zero); src points to the reply
 * buffer; data must alias src exactly. */
START_TEST(test_fill_payload_getrep)
{
	dstm_msg_t m;
	memset(&m, 0, sizeof(m));

	char buf[128];
	memset(buf, 0xCC, sizeof(buf));

	dstm_ptr_t null_ptr = {DSTM_REGIONID_INVALID, 0};
	int rc = dstm_msg_fill_payload(&m, null_ptr, buf, sizeof(buf));
	ck_assert_int_eq(rc, DSTM_SUCCESS);
	ck_assert_uint_eq(m.u.putget.size, sizeof(buf));
	/* data is a non-owning alias into the reply buffer */
	ck_assert_ptr_eq(m.data, buf);
}
END_TEST

/*******************************************************************************
 * Suite
 ******************************************************************************/

Suite *msg_suite(void)
{
	Suite *s = suite_create("msg");

	TCase *tc_fill_hdr = tcase_create("fill_hdr");
	tcase_add_test(tc_fill_hdr, test_fill_hdr_put);
	tcase_add_test(tc_fill_hdr, test_fill_hdr_getreq);
	tcase_add_test(tc_fill_hdr, test_fill_hdr_getrep);
	tcase_add_test(tc_fill_hdr, test_fill_hdr_invalid_type_max);
	tcase_add_test(tc_fill_hdr, test_fill_hdr_invalid_type_negative);
	suite_add_tcase(s, tc_fill_hdr);

	TCase *tc_fill_payload = tcase_create("fill_payload");
	tcase_add_test(tc_fill_payload, test_fill_payload_put);
	tcase_add_test(tc_fill_payload, test_fill_payload_getreq);
	tcase_add_test(tc_fill_payload, test_fill_payload_getrep);
	suite_add_tcase(s, tc_fill_payload);

	return s;
}

int main(void)
{
	int failed;
	Suite *s = msg_suite();
	SRunner *sr = srunner_create(s);
	srunner_run_all(sr, CK_VERBOSE);
	failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
