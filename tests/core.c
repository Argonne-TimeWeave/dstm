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
#include <stdlib.h>

#include "dstm.h"

START_TEST(test_version_values)
{
	ck_assert_int_eq(dstm_version_major, DSTM_VERSION_MAJOR);
	ck_assert_int_eq(dstm_version_minor, DSTM_VERSION_MINOR);
	ck_assert_int_eq(dstm_version_patch, DSTM_VERSION_PATCH);
	ck_assert_str_eq(dstm_version_revision, DSTM_VERSION_REVISION);
	ck_assert_str_eq(dstm_version_string, DSTM_VERSION_STRING);
}
END_TEST

START_TEST(test_init_null)
{
	ck_assert(!dstm_init(NULL, NULL));
	dstm_finalize();
}
END_TEST

START_TEST(test_hello_world)
{
	ck_assert(!dstm_init(NULL, NULL));





	dstm_finalize();
}
END_TEST

void setup(void)
{
	ck_assert(!dstm_init(NULL, NULL));
}

void teardown(void)
{
	dstm_finalize();
}

START_TEST(test_start_null)
{
	dstm_distance_t d = { NULL, 0 };
	ck_assert_int_eq(dstm_start(d), -DSTM_EINVAL);
}

START_TEST(test_start_id)
{
	ck_assert_int_eq(dstm_start(DSTM_DISTANCE_IDENTITY), DSTM_SUCCESS);
	ck_assert(!dstm_fini());
}

Suite *core_suite(void)
{
	Suite *s;
	TCase *tc_init;
	TCase *tc_version;
	TCase *tc_hello;
	TCase *tc_start;

	s = suite_create("core");

	tc_init = tcase_create("init");
	tcase_add_test(tc_init, test_init_null);
	suite_add_tcase(s, tc_init);

	tc_version = tcase_create("version");
	tcase_add_test(tc_version, test_version_values);
	suite_add_tcase(s, tc_version);

	tc_hello = tcase_create("hello");
	tcase_add_test(tc_hello, test_hello_world);
	suite_add_tcase(s, tc_hello);

	tc_start = tcase_create("dstm_start");
	tcase_add_checked_fixture(tc_start, setup, teardown);
	tcase_add_test(tc_start, test_start_null);
	tcase_add_test(tc_start, test_start_id);
	suite_add_tcase(s, tc_start);
	return s;
}

int main(void)
{
	int failed;
	Suite *s;
	SRunner *sr;

	s = core_suite();
	sr = srunner_create(s);

	srunner_run_all(sr, CK_VERBOSE);
	failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
