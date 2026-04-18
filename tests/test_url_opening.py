# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Comprehensive test for URL/website opening functionality.
Tests all code paths in:
  - Brain._fast_path() — URL regex matching
  - Brain._fast_path() — "open <website>" matching  
  - SystemController.open_url() — URL normalization & launching
  - SystemController.search_web() — web search
  - SystemController.open_app() — browser-related entries
  - LLM tool definitions — open_url tool schema
  
Runs offline (no subprocess, no LLM, no browser launch).
"""
import sys
import os
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Patch config for offline ───
from src.config import Config
Config.USE_GROQ = False


# ─── Patch subprocess so tests never actually launch anything ───
import subprocess
_real_Popen = subprocess.Popen

class FakePopen:
    """Records what would have been launched instead of actually launching."""
    launched = []
    def __init__(self, cmd, **kwargs):
        FakePopen.launched.append(cmd)
        self.returncode = 0
    def communicate(self, *a, **kw):
        return ("", "")

subprocess.Popen = FakePopen

class MockBrainContent:
    def __init__(self):
        pass

# Mock ollama so Brain initialization doesn't fail
import sys
sys.modules['ollama'] = MockBrainContent()

from src.system_controller import SystemController
from src.brain import Brain

# Patch Brain to ignore LLM validation
Brain._validate_llm_connection = lambda self: None

# ─── Helpers ───
PASS = 0
FAIL = 0
BUGS = []

def check(test_name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {test_name}")
    else:
        FAIL += 1
        BUGS.append((test_name, detail))
        print(f"  ✗ {test_name}  — {detail}")

# We will use Brain instance directly
try:
    brain = Brain()
except Exception as e:
    print(f"Failed to init brain: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────
#  TEST GROUP 1: fast_path — "go to / visit / navigate to" regex
# ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TEST GROUP 1: fast_path URL regex (go to / visit / navigate to)")
print("=" * 65)

url_cases = [
    # (input_text, should_match, expected_url_or_None)
    ("go to youtube.com",             True,  "youtube.com"),
    ("visit google.com",              True,  "google.com"),
    ("navigate to github.com",        True,  "github.com"),
    ("go to reddit.com/r/python",     True,  "reddit.com/r/python"),
    ("go to en.wikipedia.org",        True,  "en.wikipedia.org"),
    ("Go to Stackoverflow.com",       True,  "Stackoverflow.com"),
    # These SHOULD match but may fail due to the regex:
    ("go to https://youtube.com",     False, None),   # regex requires bare domain (no scheme)
    ("go to www.google.com",          True,  "www.google.com"),
    ("visit my-site.io",              True, "my-site.io"),   # hyphen in domain
    ("visit my_site.co",              True, "my_site.co"),   # underscore in domain
    # Negative cases (should NOT match):
    ("go to the store",               False, None),
    ("go to sleep",                   False, None),
    ("visit my grandmother",          False, None),
    ("navigate to the settings page", False, None),
]

for text, should_match, expected_url in url_cases:
    FakePopen.launched.clear()
    res = brain._fast_path(text)
    matched = (res is not None)
    launched_cmd = FakePopen.launched[-1] if FakePopen.launched else ""
    check(
        f'"{text}"',
        matched == should_match,
        f"Expected match={should_match}, got match={matched}"
    )
    if matched and should_match and expected_url:
        check(
            f'  → captured URL',
            expected_url.lower() in launched_cmd,
            f"Expected '{expected_url.lower()}' in cmd, got '{launched_cmd}'"
        )

# ──────────────────────────────────────────────────────
#  TEST GROUP 2: fast_path — "open <website>" gets routed to open_app
#  instead of open_url (the core bug)
# ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TEST GROUP 2: 'open <website>' gets routed to open_app (BUG?)")
print("=" * 65)

# When user says "open youtube.com", the fast_path FIRST checks the
# open_app regex, which matches "open (.+)" → captures "youtube.com".
# It then calls self.system.open_app("youtube.com"), which looks up
# "youtube.com" in the apps dict, finds no match, and tries the Start
# Menu search — instead of treating it as a URL.
#
# The URL regex only fires on "go to / visit / navigate to" — NOT "open".

open_website_cases = [
    "open youtube.com",
    "open google.com",
    "open https://github.com",
    "open www.reddit.com",
    "open stackoverflow.com",
    "open spotify.com",
    "open chat.openai.com",
]

for text in open_website_cases:
    FakePopen.launched.clear()
    res = brain._fast_path(text)
    launched_cmd = FakePopen.launched[-1] if FakePopen.launched else ""
    # With the fix, these should open the URL, not the app.
    # The command for URL is: start "" "https://..."
    # The command for app fallback makes multiple searches or runs: start "" "youtube.com" (without https)
    check(
        f'"{text}" — correctly routes to open_url',
        "http" in launched_cmd,
        f"Expected URL launch, got: {launched_cmd}"
    )

# ──────────────────────────────────────────────────────
#  TEST GROUP 3: SystemController.open_url() — URL normalization
# ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TEST GROUP 3: SystemController.open_url() — scheme prepending")
print("=" * 65)

sc = SystemController()

normalize_cases = [
    ("youtube.com",            "https://youtube.com"),
    ("https://youtube.com",    "https://youtube.com"),
    ("http://localhost:3000",   "http://localhost:3000"),
    ("www.google.com",         "https://www.google.com"),
    ("reddit.com/r/python",    "https://reddit.com/r/python"),
    # Edge cases
    ("",                       None),           # empty → returns error
    ("httpsite.com",           "httpsite.com"),        # scheme check
    ("HTTPS://GOOGLE.COM",    "HTTPS://GOOGLE.COM"),  # case insensitive
]

for url_input, expected in normalize_cases:
    FakePopen.launched.clear()
    result = sc.open_url(url_input)
    
    # Check what URL was actually passed to subprocess
    if expected is None:
        check(
            f'open_url("{url_input}") → returns error',
            not FakePopen.launched,
            f"Expected no launch, got: {FakePopen.launched}"
        )
        continue

    if FakePopen.launched:
        launched_cmd = FakePopen.launched[-1]
        # The command is: start "" "<url>"
        # Extract the URL from the command string
        url_in_cmd = launched_cmd.split('"')[3] if launched_cmd.count('"') >= 4 else launched_cmd
    else:
        url_in_cmd = "(nothing launched)"
    
    check(
        f'open_url("{url_input}") → cmd contains correct URL',
        expected in launched_cmd if FakePopen.launched else False,
        f"Expected URL '{expected}' in command, got: {launched_cmd if FakePopen.launched else '(nothing)'}"
    )

# ──────────────────────────────────────────────────────
#  TEST GROUP 4: SystemController.open_url() — special characters in URL
# ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TEST GROUP 4: open_url() — URLs with special characters")
print("=" * 65)

special_url_cases = [
    "https://google.com/search?q=hello world",    # space in query
    "https://example.com/path?a=1&b=2",           # ampersands
    "https://example.com/path#section",            # fragment
    "https://user:pass@example.com",               # credentials in URL
]

for url_input in special_url_cases:
    FakePopen.launched.clear()
    result = sc.open_url(url_input)
    
    if FakePopen.launched:
        launched_cmd = FakePopen.launched[-1]
        # Check if the URL appears in the command (ampersands, spaces, etc. may cause issues with shell=True)
        has_url = url_input in launched_cmd
        check(
            f'open_url("{url_input[:50]}...")',
            has_url,
            f"URL may be mangled by shell. Command: {launched_cmd}"
        )
    else:
        check(f'open_url("{url_input[:50]}...")', False, "Nothing launched")

# ──────────────────────────────────────────────────────
#  TEST GROUP 5: search_web() builds correct Google search URL
# ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TEST GROUP 5: search_web() — Google search URL construction")
print("=" * 65)

search_cases = [
    ("python tutorials",    "https://www.google.com/search?q=python%20tutorials"),
    ("hello world",         "https://www.google.com/search?q=hello%20world"),
    ("what is AI?",         "https://www.google.com/search?q=what%20is%20AI%3F"),
    ("C++ programming",     "https://www.google.com/search?q=C%2B%2B%20programming"),
]

for query, expected_url in search_cases:
    FakePopen.launched.clear()
    result = sc.search_web(query)
    
    if FakePopen.launched:
        launched_cmd = FakePopen.launched[-1]
        check(
            f'search_web("{query}")',
            expected_url in launched_cmd,
            f"Expected '{expected_url}' in command, got: {launched_cmd}"
        )
    else:
        check(f'search_web("{query}")', False, "Nothing launched")

# ──────────────────────────────────────────────────────
#  TEST GROUP 6: open_app() — browser entries
# ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TEST GROUP 6: open_app() — browser-related entries")
print("=" * 65)

browser_cases = [
    ("browser",  "start msedge"),
    ("chrome",   "start chrome"),
    ("edge",     "start msedge"),
    # These should fuzzy-match or miss:
    ("firefox",  None),          # Not in dict — goes to Start Menu search
    ("brave",    None),          # Not in dict
    ("google chrome", None),     # Not exact match
]

for app_input, expected_cmd in browser_cases:
    FakePopen.launched.clear()
    result = sc.open_app(app_input)
    
    if expected_cmd:
        if FakePopen.launched:
            launched_cmd = FakePopen.launched[0]
            check(
                f'open_app("{app_input}") → launches "{expected_cmd}"',
                expected_cmd == launched_cmd,
                f"Expected '{expected_cmd}', got '{launched_cmd}'"
            )
        else:
            check(f'open_app("{app_input}")', False, "Nothing launched")
    else:
        # Expected fallback to Start Menu search
        check(
            f'open_app("{app_input}") → fallback (Start Menu search)',
            len(FakePopen.launched) >= 1,  # PowerShell search attempt
            f"Launched: {FakePopen.launched}"
        )

# ──────────────────────────────────────────────────────
#  TEST GROUP 7: fast_path edge cases that cause misrouting
# ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TEST GROUP 7: fast_path edge cases (misrouting / false positives)")
print("=" * 65)

edge_cases = [
    # (input, should_fast_path_return_None, description)
    ("open youtube.com and play a video", True,  "complex cmd → should fall to LLM"),
    ("open chrome and go to youtube",     True,  "compound → should fall to LLM"),
    ("open the browser for me please",    False, "simple open → fast path ok"),
    ("start edge",                        False, "simple open → fast path ok"),
    ("launch chrome",                     False, "simple open → fast path ok"),
    # Tricky: "open youtube" — is it the app or the website?
    ("open youtube",                      False, "ambiguous but fast_path handles it"),
    # Long requests that accidentally match "open"
    ("open a new project called test-app on the desktop", True, "complex → should fall to LLM"),
]

for text, should_be_none, desc in edge_cases:
    FakePopen.launched.clear()
    res = brain._fast_path(text)
    result_none = (res is None)
    
    check(
        f'"{text}" — {desc}',
        result_none == should_be_none,
        f"Expected fast_path_returns_None={should_be_none}, got {result_none}"
    )

# ──────────────────────────────────────────────────────
#  TEST GROUP 8: "open <url>" — the missing code path
# ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TEST GROUP 8: Missing 'open <url>' code path in fast_path")
print("=" * 65)

# The fast_path has:
#   1. "open/start/launch <app>" → open_app()
#   2. "go to/visit/navigate to <url>" → open_url()
# 
# But there's NO handler for "open youtube.com" → open_url().
# "open youtube.com" hits case 1 and routes to open_app(),
# which doesn't know what to do with a URL.

missing_path_cases = [
    "open youtube.com",
    "open google.com",
    "open github.com/trending",
    "open reddit.com",
    "open https://chat.openai.com",
]

for text in missing_path_cases:
    FakePopen.launched.clear()
    res = brain._fast_path(text)
    launched_cmd = FakePopen.launched[-1] if FakePopen.launched else ""
    check(
        f'"{text}" — handled correctly',
        "http" in launched_cmd,
        f"Expected URL launch, got: {launched_cmd}"
    )

# ──────────────────────────────────────────────────────
#  SUMMARY
# ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print("=" * 65)

if BUGS:
    print("\n🐛 BUGS FOUND:")
    print("-" * 65)
    for i, (name, detail) in enumerate(BUGS, 1):
        print(f"\n  Bug #{i}: {name}")
        print(f"    Detail: {detail}")
    
    print("\n" + "-" * 65)
    print("\n📋 SUMMARY OF KEY BUGS:")
    print("""
  1. CRITICAL: "open youtube.com" routes to open_app() instead of open_url().
     The fast_path "open/start/launch" regex (line ~450 in brain.py) fires
     BEFORE the URL regex. Since "open" is not in the URL regex trigger words
     ("go to/visit/navigate to"), URLs with "open" prefix go to open_app(),
     which searches the Start Menu and fails.
     
     FIX: In _fast_path(), after matching the "open" regex, check if the
     captured text looks like a URL (contains a dot + TLD). If so, route
     to self.system.open_url() instead of self.system.open_app().

  2. MINOR: open_url("httpsite.com") doesn't prepend "https://" because 
     the check is `url.startswith("http")` — which matches "httpsite.com" 
     as already having a scheme. Should check for "http://" or "https://" 
     with the trailing "://".

  3. MINOR: URLs with special characters (spaces, ampersands) may be mangled
     when passed through `shell=True` in subprocess.Popen. Consider using
     webbrowser.open() instead of subprocess with `start "" "<url>"`.
     
  4. MINOR: "open https://youtube.com" — the URL regex won't match because
     it requires bare domain. The open_app regex catches it and sends
     "https://youtube.com" to open_app(), which can't handle it.

  5. MINOR: open_url("") prepends "https://" to empty string → "https://".
     Should validate that URL is non-empty.
""")

# Restore subprocess
subprocess.Popen = _real_Popen
