CODEACT_TESTGEN_PROMPT = """
Your ONLY goal is to write a test that **reproduces the bug** described below. The test should FAIL on the current code and would PASS after the patch is applied.

Place your test in: {test_file}

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
2. **Write a minimal test** - Create a test that exercises the changed code path
3. **Verify it fails** - Run: <execute_bash> {coverage_command} </execute_bash>
4. **Iterate if needed** - If the test doesn't fail, analyze why and adjust
5. **Exit when done** - Once you have a FAILING test: <execute_bash> exit </execute_bash>

## CRITICAL RULES:

- **Quality over quantity** - ONE good failing test is better than many random tests
- **Focus on the patch** - Your test must exercise the exact lines changed in the patch
- **Must fail before patch** - Verify the test actually fails on current code
- **Do NOT add extra tests** - Only write tests that reproduce THIS specific bug
- **Do NOT aim for coverage** - Coverage is not the goal, bug reproduction is
- **Do NOT modify source** - Only write to {test_file}

## CODE TO TEST:

<START_OF_CODE>
{code_src}
<END_OF_CODE>

## NOTES:

- Django projects: Use `from django.test import SimpleTestCase` and class-based tests
- Fix any import errors before exiting
- Use local imports, not global library imports
- Replace {test_file} if you need to revise your test suite

**SUCCESS = At least ONE test that FAILS on the current code**
"""

CODEACT_TESTGEN_PROMPT_ITERATE = """
Your ONLY goal is to improve the test suite at {test_file} to reproduce the bug. Focus on FAIL-TO-PASS tests only.

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
2. **Check if bug is reproduced** - Do any tests fail that would pass with the patch?
3. **If NO failing tests** - Add tests that exercise the changed code paths in the patch
4. **If tests don't run** - Delete {test_file} and create a new test file from scratch
5. **Exit when done** - Once you have FAILING test(s): <execute_bash> exit </execute_bash>

## CRITICAL RULES:

- **Focus on fail-to-pass** - Only write tests that would fail now and pass after the patch
- **Study the patch** - Your tests must exercise the exact lines changed
- **Verify failures** - Run tests to confirm they actually fail
- **Do NOT aim for coverage** - Coverage is not the goal, bug reproduction is
- **Do NOT write passing tests** - Only tests that reproduce the bug matter
- **Do NOT modify source** - Only write to {test_file}

## CODE TO TEST:

<START_OF_CODE>
{code_src}
<END_OF_CODE>

## NOTES:

- Django projects: Use `from django.test import SimpleTestCase` and class-based tests
- Fix any import errors before exiting
- Use local imports, not global library imports
- Replace {test_file} entirely if you need to rewrite the test suite

**SUCCESS = At least ONE test that FAILS on the current code and targets the patch**
"""
