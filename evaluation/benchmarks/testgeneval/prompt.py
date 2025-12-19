CODEACT_TESTGEN_PROMPT = """
Your ONLY goal is to write tests that **reproduce the bug** described below.

CRITICAL: Each test MUST be a FAIL-TO-PASS test:
- The test MUST FAIL when run on the current buggy code (before patch)
- The test MUST PASS when run after the patch is applied (bug fixed)

IMPORTANT: Create a NEW test file at: {test_file}
This is a SEPARATE file - do NOT modify existing test files.
You must write AT LEAST 3 DIFFERENT tests that reproduce this bug from different angles.

## THE BUG TO REPRODUCE:

PR / ISSUE CONTEXT:
- PR Title: {pr_title}
- PR Description: {pr_description}
- Issue: {issue_text}
- Hints: {hints_text}

## THE FIX (study this carefully):
<START_OF_PATCH>
{patch}
<END_OF_PATCH>

The patch shows EXACTLY what was changed to fix the bug. Your test should exercise these specific lines.

{existing_tests}

## YOUR TASK:

1. **Study the patch** - Understand what code was changed and why
2. **Write at least 3 tests** - Create different tests that exercise the changed code path from various angles
3. **Verify they FAIL on current code** - Run: <execute_bash> {coverage_command} </execute_bash>
   - All tests MUST fail because the bug still exists
   - If a test passes, it's NOT reproducing the bug - rewrite it
4. **Ensure they would PASS after patch** - Each test should target the exact bug that the patch fixes
5. **Iterate if needed** - If the tests don't fail, analyze why and adjust
6. **Exit when done** - Once you have at least 3 FAIL-TO-PASS tests: <execute_bash> exit </execute_bash>

## CRITICAL RULES:

- **Create NEW file** - Write to {test_file} (a separate file, NOT the existing tests)
- **Write at least 3 tests** - Each test should approach the bug from a different angle or test different edge cases
- **Diverse test scenarios** - Tests should cover different inputs, edge cases, or aspects of the bug
- **FAIL-TO-PASS requirement** - EVERY test MUST:
  * FAIL on the current buggy code (before patch is applied)
  * PASS after the patch is applied (when bug is fixed)
  * If a test passes now, it's useless - delete it and write a real bug-reproducing test
- **Focus on the patch** - Your tests must exercise the exact lines changed in the patch
- **Verify failures NOW** - Run the tests and confirm they all fail on current code
- **Quality matters** - Write meaningful tests that genuinely reproduce the bug, not redundant ones
- **Do NOT aim for coverage** - Coverage is not the goal, bug reproduction from multiple angles is
- **Do NOT modify source** - Only write to the NEW test file {test_file}

## CODE TO TEST:

<START_OF_CODE>
{code_src}
<END_OF_CODE>

## NOTES:

- Django projects: Use `from django.test import SimpleTestCase` and class-based tests
- Fix any import errors before exiting
- Use local imports, not global library imports
- Overwrite {test_file} completely if you need to revise your test suite

**SUCCESS = At least THREE different FAIL-TO-PASS tests:**
- Each test FAILS on the current buggy code
- Each test would PASS after the patch is applied
- Each test targets different aspects of the bug
"""

CODEACT_TESTGEN_PROMPT_ITERATE = """
Your ONLY goal is to improve the test suite at {test_file} to reproduce the bug.

CRITICAL: Each test MUST be a FAIL-TO-PASS test:
- The test MUST FAIL when run on the current buggy code (before patch)
- The test MUST PASS when run after the patch is applied (bug fixed)

IMPORTANT: {test_file} is a SEPARATE test file (with _openhands suffix) - do NOT modify existing tests.
You must have AT LEAST 3 DIFFERENT tests that reproduce this bug from different angles.

## THE BUG TO REPRODUCE:

PR / ISSUE CONTEXT:
- PR Title: {pr_title}
- PR Description: {pr_description}
- Issue: {issue_text}
- Hints: {hints_text}

## THE FIX (study this carefully):
<START_OF_PATCH>
{patch}
<END_OF_PATCH>

The patch shows EXACTLY what was changed to fix the bug. Your tests should exercise these specific lines.

{existing_tests}

## YOUR TASK:

1. **Run existing tests first**: <execute_bash> {coverage_command} </execute_bash>
2. **Check FAIL-TO-PASS status** - Do you have at least 3 tests that:
   - FAIL on current buggy code?
   - Would PASS after the patch is applied?
3. **Remove any passing tests** - If a test passes now, it's NOT reproducing the bug - delete it
4. **If fewer than 3 FAIL-TO-PASS tests** - Add more tests that exercise the changed code paths from different angles
5. **If tests don't run** - Delete {test_file} and create a new test file from scratch with at least 3 tests
6. **Verify all tests FAIL** - Confirm every test fails because the bug still exists
7. **Exit when done** - Once you have at least 3 FAIL-TO-PASS tests: <execute_bash> exit </execute_bash>

## CRITICAL RULES:

- **Separate file only** - {test_file} is a NEW file with _openhands suffix, NOT existing tests
- **Write at least 3 tests** - Each test should approach the bug from a different angle or test different edge cases
- **Diverse test scenarios** - Tests should cover different inputs, edge cases, or aspects of the bug
- **FAIL-TO-PASS requirement** - EVERY test MUST:
  * FAIL on the current buggy code (before patch is applied)
  * PASS after the patch is applied (when bug is fixed)
  * If a test passes now, DELETE it and write a real bug-reproducing test
- **Study the patch** - Your tests must exercise the exact lines changed
- **Verify failures NOW** - Run the tests and confirm they all fail on current code
- **Do NOT aim for coverage** - Coverage is not the goal, bug reproduction from multiple angles is
- **Do NOT write passing tests** - Passing tests are useless for bug reproduction - DELETE them
- **Do NOT modify source** - Only write to the NEW test file {test_file}

## CODE TO TEST:

<START_OF_CODE>
{code_src}
<END_OF_CODE>

## NOTES:

- Django projects: Use `from django.test import SimpleTestCase` and class-based tests
- Fix any import errors before exiting
- Use local imports, not global library imports
- Overwrite {test_file} entirely if you need to rewrite the test suite

**SUCCESS = At least THREE different FAIL-TO-PASS tests:**
- Each test FAILS on the current buggy code
- Each test would PASS after the patch is applied
- Each test targets different aspects of the bug
- NO tests pass on the current buggy code
"""
