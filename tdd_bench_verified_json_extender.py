#!/usr/bin/env python3
"""
Script to extend SWE-bench instance data by fetching missing information from GitHub.

Required fields for the output:
- repo: Repo slug (e.g., django/django)
- base_commit: SHA of the code snapshot
- version: Dataset version tag
- instance_id: Unique task ID
- id: Same as instance_id (required duplicate)
- patch: Code diff for the fix
- test_patch: Test diff that accompanied the fix
- preds_context: Dict for extra context (issue text, hints, difficulty)
- code_src: Full contents of the code file under test
- test_src: Contents of existing tests (can be empty)
- code_file: Path to the code under test
- test_file: Path where tests should be written
- local_imports: List of import hints
- baseline_covs: Dict of baseline coverage metrics (can be {})
"""

import json
import re
import sys
import subprocess
from pathlib import Path
from typing import Any, Optional

# Try to import requests, but make it optional
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def get_github_file_content(repo: str, commit: str, file_path: str, token: Optional[str] = None) -> Optional[str]:
    """Fetch file content from GitHub at a specific commit."""
    if not REQUESTS_AVAILABLE:
        print("Warning: 'requests' module not available. Trying curl fallback...")
        return get_github_file_content_curl(repo, commit, file_path, token)
    
    # Try raw.githubusercontent.com first (no API rate limits)
    raw_url = f"https://raw.githubusercontent.com/{repo}/{commit}/{file_path}"
    try:
        response = requests.get(raw_url, timeout=30)
        if response.status_code == 200:
            return response.text
    except requests.exceptions.RequestException as e:
        print(f"Warning: raw.githubusercontent.com failed: {e}")
    
    # Fallback to API
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}?ref={commit}"
    headers = {"Accept": "application/vnd.github.v3.raw"}
    if token:
        headers["Authorization"] = f"token {token}"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Warning: Could not fetch {file_path} at {commit}: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Warning: GitHub API request failed: {e}")
        return None


def get_github_file_content_curl(repo: str, commit: str, file_path: str, token: Optional[str] = None) -> Optional[str]:
    """Fallback using curl for fetching GitHub content."""
    raw_url = f"https://raw.githubusercontent.com/{repo}/{commit}/{file_path}"
    try:
        cmd = ["curl", "-sL", raw_url]
        if token:
            cmd.extend(["-H", f"Authorization: token {token}"])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout and "404: Not Found" not in result.stdout:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Warning: curl fallback failed: {e}")
    return None


def extract_code_file_from_patch(patch: str) -> Optional[str]:
    """Extract the code file path from the patch diff.
    
    Returns the first non-test Python file found in the patch.
    """
    # Look for all files in the patch
    # Pattern: diff --git a/path/to/file.py b/path/to/file.py
    matches = re.findall(r'diff --git a/(.+\.py) b/\1', patch)
    
    # Filter out test files to get the main code file
    for file_path in matches:
        if '/test' not in file_path.lower() and not file_path.lower().startswith('test'):
            return file_path
    
    # If all are test files or no matches, return first match or None
    return matches[0] if matches else None


def extract_test_file_from_test_patch(test_patch: str) -> Optional[str]:
    """Extract the test file path from the test patch diff.
    
    Returns the first test Python file found in the test patch.
    """
    matches = re.findall(r'diff --git a/(.+\.py) b/\1', test_patch)
    
    # Prefer files that look like test files
    for file_path in matches:
        if '/test' in file_path.lower() or file_path.lower().startswith('test'):
            return file_path
    
    # Return first match if no obvious test file found
    return matches[0] if matches else None


def extract_all_files_from_patch(patch: str) -> list[str]:
    """Extract all file paths from a patch."""
    matches = re.findall(r'diff --git a/(.+) b/\1', patch)
    return list(dict.fromkeys(matches))  # Remove duplicates while preserving order


def extract_local_imports(code_src: str, repo: Optional[str] = None) -> list[str]:
    """Extract local imports from the code source.
    
    Args:
        code_src: The source code to analyze
        repo: Repository name (e.g., 'astropy/astropy') to help identify local imports
    """
    imports = set()
    
    # Match "from X import Y" patterns
    from_imports = re.findall(r'^from\s+([\w.]+)\s+import', code_src, re.MULTILINE)
    imports.update(from_imports)
    
    # Match "import X" patterns  
    direct_imports = re.findall(r'^import\s+([\w.]+)', code_src, re.MULTILINE)
    imports.update(direct_imports)
    
    # Standard library modules to exclude
    stdlib_modules = {
        'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore',
        'atexit', 'base64', 'bdb', 'binascii', 'binhex', 'bisect', 'builtins',
        'bz2', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd', 'code',
        'codecs', 'codeop', 'collections', 'colorsys', 'compileall', 'concurrent',
        'configparser', 'contextlib', 'contextvars', 'copy', 'copyreg', 'cProfile',
        'crypt', 'csv', 'ctypes', 'curses', 'dataclasses', 'datetime', 'dbm',
        'decimal', 'difflib', 'dis', 'distutils', 'doctest', 'email', 'encodings',
        'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput', 'fnmatch',
        'fractions', 'ftplib', 'functools', 'gc', 'getopt', 'getpass', 'gettext',
        'glob', 'grp', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http',
        'imaplib', 'imghdr', 'imp', 'importlib', 'inspect', 'io', 'ipaddress',
        'itertools', 'json', 'keyword', 'lib2to3', 'linecache', 'locale', 'logging',
        'lzma', 'mailbox', 'mailcap', 'marshal', 'math', 'mimetypes', 'mmap',
        'modulefinder', 'multiprocessing', 'netrc', 'nis', 'nntplib', 'numbers',
        'operator', 'optparse', 'os', 'ossaudiodev', 'parser', 'pathlib', 'pdb',
        'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform', 'plistlib',
        'poplib', 'posix', 'posixpath', 'pprint', 'profile', 'pstats', 'pty',
        'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri', 'random',
        're', 'readline', 'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched',
        'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil', 'signal',
        'site', 'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'spwd',
        'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct',
        'subprocess', 'sunau', 'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny',
        'tarfile', 'telnetlib', 'tempfile', 'termios', 'test', 'textwrap', 'threading',
        'time', 'timeit', 'tkinter', 'token', 'tokenize', 'trace', 'traceback',
        'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types', 'typing', 'unicodedata',
        'unittest', 'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref',
        'webbrowser', 'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc',
        'zipapp', 'zipfile', 'zipimport', 'zlib', '_thread', '__future__'
    }
    
    # Common third-party packages to exclude
    third_party = {
        'numpy', 'np', 'scipy', 'matplotlib', 'plt', 'pandas', 'pd', 'sklearn',
        'torch', 'tensorflow', 'tf', 'keras', 'cv2', 'PIL', 'requests', 'flask',
        'django', 'pytest', 'nose', 'setuptools', 'pkg_resources', 'six', 'yaml',
        'toml', 'tqdm', 'click', 'jinja2', 'sqlalchemy', 'celery', 'redis',
        'boto3', 'botocore', 'aws', 'google', 'azure'
    }
    
    # Determine project package name from repo
    project_package = None
    if repo:
        # Extract package name from repo (e.g., 'astropy/astropy' -> 'astropy')
        parts = repo.split('/')
        if len(parts) >= 2:
            project_package = parts[1].lower().replace('-', '_')
    
    local_imports = []
    for imp in imports:
        root_module = imp.split('.')[0].lower()
        
        # Skip stdlib and third-party
        if root_module in stdlib_modules or root_module in third_party:
            continue
        
        # If we know the project package, prioritize imports from it
        if project_package:
            if root_module == project_package or imp.lower().startswith(project_package + '.'):
                local_imports.append(imp)
        else:
            # Without repo context, include non-stdlib/non-third-party imports
            local_imports.append(imp)
    
    return sorted(set(local_imports))


def extend_instance_data(instance: dict, github_token: Optional[str] = None) -> dict:
    """Extend instance data with missing fields fetched from GitHub."""
    
    # Copy existing data
    extended = instance.copy()
    
    # Ensure 'id' field exists (duplicate of instance_id)
    if 'id' not in extended and 'instance_id' in extended:
        extended['id'] = extended['instance_id']
    
    repo = extended.get('repo')
    base_commit = extended.get('base_commit')
    patch = extended.get('patch', '')
    test_patch = extended.get('test_patch', '')
    
    if not repo or not base_commit:
        print("Error: 'repo' and 'base_commit' are required fields")
        return extended
    
    # Extract code_file from patch if not present
    if 'code_file' not in extended and patch:
        code_file = extract_code_file_from_patch(patch)
        if code_file:
            extended['code_file'] = code_file
            print(f"Extracted code_file: {code_file}")
    
    # Extract test_file from test_patch if not present
    if 'test_file' not in extended and test_patch:
        test_file = extract_test_file_from_test_patch(test_patch)
        if test_file:
            extended['test_file'] = test_file
            print(f"Extracted test_file: {test_file}")
    
    # Fetch code_src from GitHub if not present
    if 'code_src' not in extended and 'code_file' in extended:
        print(f"Fetching code source from GitHub: {extended['code_file']}")
        code_src = get_github_file_content(repo, base_commit, extended['code_file'], github_token)
        if code_src:
            extended['code_src'] = code_src
            print(f"Fetched code_src ({len(code_src)} chars)")
    
    # Fetch test_src from GitHub if not present
    if 'test_src' not in extended and 'test_file' in extended:
        print(f"Fetching test source from GitHub: {extended['test_file']}")
        test_src = get_github_file_content(repo, base_commit, extended['test_file'], github_token)
        if test_src:
            extended['test_src'] = test_src
            print(f"Fetched test_src ({len(test_src)} chars)")
        else:
            # Test file might not exist at base_commit (new tests)
            extended['test_src'] = ""
            print("Test file not found at base_commit (may be new), setting test_src to empty")
    
    # Extract local_imports from code_src if not present
    if 'local_imports' not in extended and 'code_src' in extended:
        local_imports = extract_local_imports(extended['code_src'], repo)
        extended['local_imports'] = local_imports
        print(f"Extracted local_imports: {local_imports}")
    
    # Initialize baseline_covs if not present
    if 'baseline_covs' not in extended:
        extended['baseline_covs'] = {}
        print("Initialized baseline_covs to empty dict")
    
    # Initialize preds_context if not present
    if 'preds_context' not in extended:
        preds_context = {}
        # Add problem_statement to context if available
        if 'problem_statement' in extended:
            preds_context['issue_text'] = extended['problem_statement']
        if 'hints_text' in extended:
            preds_context['hints'] = extended['hints_text']
        if 'difficulty' in extended:
            preds_context['difficulty'] = extended['difficulty']
        extended['preds_context'] = preds_context
        print(f"Created preds_context with keys: {list(preds_context.keys())}")
    
    return extended


def process_json_file(input_path: str, output_path: Optional[str] = None, github_token: Optional[str] = None):
    """Process a JSON file and extend all instances."""
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Handle both single instance and list of instances
    if isinstance(data, list):
        extended_data = []
        total = len(data)
        for i, instance in enumerate(data, 1):
            instance_id = instance.get('instance_id', f'instance_{i}')
            print(f"\n[{i}/{total}] Processing: {instance_id}")
            print("-" * 50)
            extended = extend_instance_data(instance, github_token)
            extended_data.append(extended)
        print(f"\n{'='*60}")
        print(f"Completed processing {total} instances")
        print(f"{'='*60}")
    else:
        extended_data = extend_instance_data(data, github_token)
    
    # Determine output path
    if output_path is None:
        input_p = Path(input_path)
        output_path = str(input_p.parent / f"{input_p.stem}_extended{input_p.suffix}")
    
    with open(output_path, 'w') as f:
        json.dump(extended_data, f, indent=2)
    
    print(f"\nExtended data written to: {output_path}")
    return extended_data


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extend SWE-bench instance data with GitHub information')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('-o', '--output', help='Output JSON file path (default: input_extended.json)')
    parser.add_argument('-t', '--token', help='GitHub API token (optional, for higher rate limits)')
    
    args = parser.parse_args()
    
    process_json_file(args.input_file, args.output, args.token)


if __name__ == '__main__':
    # Example usage with the provided data
    example_data = {
        "repo": "astropy/astropy",
        "instance_id": "astropy__astropy-12907",
        "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
        "patch": "diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py\n--- a/astropy/modeling/separable.py\n+++ b/astropy/modeling/separable.py\n@@ -242,7 +242,7 @@ def _cstack(left, right):\n         cright = _coord_matrix(right, 'right', noutp)\n     else:\n         cright = np.zeros((noutp, right.shape[1]))\n-        cright[-right.shape[0]:, -right.shape[1]:] = 1\n+        cright[-right.shape[0]:, -right.shape[1]:] = right\n \n     return np.hstack([cleft, cright])\n \n",
        "test_patch": "diff --git a/astropy/modeling/tests/test_separable.py b/astropy/modeling/tests/test_separable.py\n--- a/astropy/modeling/tests/test_separable.py\n+++ b/astropy/modeling/tests/test_separable.py\n@@ -28,6 +28,13 @@\n p1 = models.Polynomial1D(1, name='p1')\n \n \n+cm_4d_expected = (np.array([False, False, True, True]),\n+                  np.array([[True,  True,  False, False],\n+                            [True,  True,  False, False],\n+                            [False, False, True,  False],\n+                            [False, False, False, True]]))\n+\n+\n compound_models = {\n     'cm1': (map3 & sh1 | rot & sh1 | sh1 & sh2 & sh1,\n             (np.array([False, False, True]),\n@@ -52,7 +59,17 @@\n     'cm7': (map2 | p2 & sh1,\n             (np.array([False, True]),\n              np.array([[True, False], [False, True]]))\n-            )\n+            ),\n+    'cm8': (rot & (sh1 & sh2), cm_4d_expected),\n+    'cm9': (rot & sh1 & sh2, cm_4d_expected),\n+    'cm10': ((rot & sh1) & sh2, cm_4d_expected),\n+    'cm11': (rot & sh1 & (scl1 & scl2),\n+             (np.array([False, False, True, True, True]),\n+              np.array([[True,  True,  False, False, False],\n+                        [True,  True,  False, False, False],\n+                        [False, False, True,  False, False],\n+                        [False, False, False, True,  False],\n+                        [False, False, False, False, True]]))),\n }\n \n \n",
        "problem_statement": "Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels\nConsider the following model:\r\n\r\n```python\r\nfrom astropy.modeling import models as m\r\nfrom astropy.modeling.separable import separability_matrix\r\n\r\ncm = m.Linear1D(10) & m.Linear1D(5)\r\n```\r\n\r\nIt's separability matrix as you might expect is a diagonal:\r\n\r\n```python\r\n>>> separability_matrix(cm)\r\narray([[ True, False],\r\n       [False,  True]])\r\n```\r\n\r\nIf I make the model more complex:\r\n```python\r\n>>> separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))\r\narray([[ True,  True, False, False],\r\n       [ True,  True, False, False],\r\n       [False, False,  True, False],\r\n       [False, False, False,  True]])\r\n```\r\n\r\nThe output matrix is again, as expected, the outputs and inputs to the linear models are separable and independent of each other.\r\n\r\nIf however, I nest these compound models:\r\n```python\r\n>>> separability_matrix(m.Pix2Sky_TAN() & cm)\r\narray([[ True,  True, False, False],\r\n       [ True,  True, False, False],\r\n       [False, False,  True,  True],\r\n       [False, False,  True,  True]])\r\n```\r\nSuddenly the inputs and outputs are no longer separable?\r\n\r\nThis feels like a bug to me, but I might be missing something?\n",
        "hints_text": "",
        "created_at": "2022-03-03T15:14:54Z",
        "version": "4.3",
        "environment_setup_commit": "298ccb478e6bf092953bca67a3d29dc6c35f6752",
        "difficulty": "15 min - 1 hour"
    }
    
    if len(sys.argv) > 1:
        main()
    else:
        # Run example
        print("Running example with provided data...\n")
        extended = extend_instance_data(example_data)
        
        print("\n" + "="*60)
        print("Extended instance data:")
        print("="*60)
        
        # Print without code_src and test_src for readability
        display_data = {k: v for k, v in extended.items() if k not in ['code_src', 'test_src']}
        display_data['code_src'] = f"<{len(extended.get('code_src', ''))} chars>" if extended.get('code_src') else None
        display_data['test_src'] = f"<{len(extended.get('test_src', ''))} chars>" if extended.get('test_src') else None
        
        print(json.dumps(display_data, indent=2))
        
        # Save full output
        with open('/home/claude/extended_example.json', 'w') as f:
            json.dump(extended, f, indent=2)
        print("\nFull output saved to: /home/claude/extended_example.json")
