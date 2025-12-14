CODEACT_TESTGEN_PROMPT_OLD = """Your goal is to generate a high-quality test suite (at least 20+ passing tests) for the code file: {code_file}. Output the test suite at {test_file}\n'

[current directory: /workspace/{workspace_dir_name}]

IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP

IMPORTANT: Follow instructions, if you have < 80 tests you should generate more tests rather than trying to fix the ones you have.

IMPORTANT: Code file to test:
```python
{code_src}
```

Here are additional imports that you may need:
{imports}

Look at code dependencies (NOT {code_file} since you already have contents) and test files you need context for to write a complete test suite.

Aim for 20+ test functions with asserts. Do not hestitate to use the Python interpreter to understand the input output behavior of the code you are testing.

Output your test suite at {test_file}. Each unit test must be a function starting with test_. Include all your test imports and setup before your first test. Do not include a main method to run the tests. Make sure to make it as comprehensive as possible, try to execute all the methods you saw.

When you think you've successfully generated a test suite, run it on for the current project using {coverage_command}.

If you have few tests GENERATE MORE TESTS rather than trying to fix the ones you have (it is possible to filter out failing tests later).

Then run coverage report -m --include {code_file} to see how well your test suite covers the code under test.

When you are trying to improve coverage pick a part of the code that is not covered (indicated by lines on coverage report), examine the code and then
try to generate a test for it. Feel free to use a code interpreter to understand the input output behavior. ONLY add tests
not remove them.

If you are unable to see passing and failing tests, FIX YOUR IMPORTS to use the same style as other test files.

You should NOT modify any existing test case files. You SHOULD add new test in a NEW file to reproduce the issue.

You should NEVER use web browsing or any other web-based tools.

You should NEVER install new packages, use existing packages only.

You should ALWAYS use the default Python interpreter available in the <execute_bash> environment to run code related to the provided issue and/or repository.

You should ALWAYS use local imports DO NOT import the general library.

When you think you have a fully adequate test suite, please run the following command: <execute_bash> exit </execute_bash>.
"""

CODEACT_TESTGEN_PROMPT = """
Your goal is to write a test suite for the code below, specifically designing it to **reproduce the issue** described in the context (Fail-to-Pass).

Place your test suite in a new file named {test_file}.

PR / ISSUE CONTEXT:
- PR Title: {pr_title}
- PR Description: {pr_description}
- Issue: {issue_text}
- Hints: {hints_text}

{existing_tests}

IMPORTANT REQUIREMENTS:
1. **Reproduce the Issue**: Your PRIMARY and ONLY goal is to write a test case that fails on the current code but represents the intended behavior (as described in the issue).
2. **Self-Contained**: Use only the provided code snippet and standard/local imports. Do NOT use external resources.
3. **Validation**: Run the tests using the command below to confirm you have at least one FAILING test (reproducing the issue).
   <execute_bash> {coverage_command} </execute_bash>
4. **No Modifications**: Do NOT modify existing source code or test files. Only write to {test_file}.
5. **Iterate**: If the issue is not reproduced, analyze the failure and add more targeted tests.
6. **Finalize**: When you have satisfied the requirements, exit.
   <execute_bash> exit </execute_bash>

Below is the **complete code snippet** to test:

<START_OF_CODE>
{code_src}
<END_OF_CODE>

NOTE: if you are testing django, you must use from django.test import SimpleTestCase and class based tests (i.e. class TestSomething(SimpleTestCase)).
NOTE: if there is an error executing tests you MUST fix it before exiting. DO NOT install new packages.
NOTE: if outputting a revised test suite REPLACE {test_file} with the revised suite

**Output the final test suite** for {test_file} in a single code block. MAKE SURE you run the tests and see at least one FAILURE (reproduction) before exiting.
"""

CODEACT_TESTGEN_PROMPT_ITERATE = """
Your goal is to improve the test suite at {test_file} to achieve **broad-coverage** of the code below.

First run the test suite.

If no tests run, then remove {test_file} and create {test_file} with a new suite.

Otherwise, improve it aiming to improve code coverage.

PR / ISSUE CONTEXT (use this to craft fail-to-pass tests first):
- PR Title: {pr_title}
- PR Description: {pr_description}
- Issue: {issue_text}
- Hints: {hints_text}

{existing_tests}

IMPORTANT REQUIREMENTS:
1. Run the provided test command (RUN THIS FIRST) to observe failures/passes (no coverage tooling needed):
   <execute_bash> {coverage_command} </execute_bash>
   If the issue is not reproduced, add new fail-to-pass tests targeting it.
2. **No external help or resources**—use only the snippet below.
3. **Focus on breadth over depth**: cover all major functions, classes, and code paths early to minimize iterations.
4. Each test function must use `assert` to verify behavior.
5. Include only necessary imports (standard library or local).
6. Do **not** modify other test files in the repository. No `main()` or other non-test code.
7. Produce **at least 20 test functions**; if coverage is lacking, add more tests rather than removing or changing existing ones.
8. When you're satisfied, finalize by running:
   <execute_bash> exit </execute_bash>

Below is the **complete code snippet** to test:

<START_OF_CODE>
{code_src}
<END_OF_CODE>

NOTE: if you are testing django, you must use from django.test import SimpleTestCase and class based tests (i.e. class TestSomething(SimpleTestCase)).
NOTE: if there is an error executing tests you MUST fix it before exiting. DO NOT install new packages.
NOTE: if outputting a revised test suite REPLACE {test_file} with the revised suite

**Output the final test suite** (20+ tests) for {test_file} in a single code block, no extra commentary. MAKE SURE you run the tests and ensure you can see which tests passed and failed BEFORE exiting.
"""
