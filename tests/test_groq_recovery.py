"""Test that the Groq failed_generation recovery logic works."""
import re, json, ast, sys

# Simulate the exact error string from the user's logs
error_str = """Error code: 400 - {'error': {'message': "Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.", 'type': 'invalid_request_error', 'code': 'tool_use_failed', 'failed_generation': '<function=open_app{"app_name": "chrome"}</function>\n'}}"""

print("=== Test 1: ast.literal_eval extraction ===")
err_content = None
try:
    err_body_match = re.search(r"(\{.*\})\s*$", error_str, re.DOTALL)
    print(f"Body regex matched: {bool(err_body_match)}")
    if err_body_match:
        err_data = ast.literal_eval(err_body_match.group(1))
        err_content = err_data.get('error', {}).get('failed_generation', '')
        print(f"err_content: {repr(err_content)}")
except Exception as e:
    print(f"ast.literal_eval failed: {e}")

if not err_content:
    print("\n=== Test 1b: Regex fallback ===")
    fg_match = re.search(r"'failed_generation':\s*'(.*?)'(?:\s*\})", error_str, re.DOTALL)
    if fg_match:
        err_content = fg_match.group(1)
        print(f"Regex fallback err_content: {repr(err_content)}")
    else:
        print("Regex fallback also failed!")

print(f"\n=== Test 2: <function=...> pattern matching ===")
if err_content:
    # This is the FIXED regex with \s* (zero or more spaces)
    func_match = re.search(r'<function=(\w+)\s*(.*?)\s*</function>', err_content, re.DOTALL)
    print(f"Function regex matched: {bool(func_match)}")
    if func_match:
        tool_name = func_match.group(1)
        args_str = func_match.group(2).strip()
        print(f"tool_name: {tool_name}")
        print(f"args_str: {repr(args_str)}")
        
        args_match = re.search(r'(\{.*\})', args_str, re.DOTALL)
        if args_match:
            args = json.loads(args_match.group(1))
            print(f"Parsed args: {args}")
            print("\n=== RESULT: SUCCESS ===")
            print(f"Recovered tool call: {tool_name}({args})")
        else:
            print("FAIL: Could not find JSON in args_str")
    else:
        print("FAIL: Function regex did not match!")
        # Debug: test the OLD regex too
        old_match = re.search(r'<function=(\w+)\s+(.*?)</function>', err_content, re.DOTALL)
        print(f"Old regex (with \\s+) matched: {bool(old_match)}")
else:
    print("FAIL: No err_content extracted")

# Test with the other format too (with space)
print("\n=== Test 3: Format WITH space ===")
err_content2 = '<function=open_url {"url": "https://www.youtube.com"} </function>'
func_match2 = re.search(r'<function=(\w+)\s*(.*?)\s*</function>', err_content2, re.DOTALL)
if func_match2:
    args2 = json.loads(re.search(r'(\{.*\})', func_match2.group(2)).group(1))
    print(f"Recovered: {func_match2.group(1)}({args2})")
    print("SUCCESS")
else:
    print("FAIL")
