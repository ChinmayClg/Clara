"""Quick verification of all 7 Ollama brain fixes."""
import os, sys, inspect
os.environ['GEMINI_API_KEY'] = 'placeholder'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.brain import Brain

brain_src = inspect.getsource(Brain)
init_src = inspect.getsource(Brain.__init__)
fp_src = inspect.getsource(Brain._fast_path)
retry_src = inspect.getsource(Brain._call_ollama_with_retry)
stream_src = inspect.getsource(Brain._stream_response)
pc_src = inspect.getsource(Brain.process_command)

tests = [
    ("model alias", "self.model = self.primary_model" in init_src),
    ("prompt trimmed", "You are a desktop assistant" in init_src),
    ("fast-path no search call", "system.search_web" not in fp_src),
    ("temperature in chat", "temperature" in retry_src and "temperature" in stream_src),
    ("search_web restrictive", "NEVER use for coding" in brain_src),
    ("re-prompt no all_results guard", "all_results and content.strip" not in pc_src),
    ("loop guard", "last_call_signature" in pc_src and "repeated_call_count" in pc_src),
]

all_pass = True
for name, result in tests:
    status = "PASS" if result else "FAIL"
    if not result:
        all_pass = False
    print(f"  {status}: {name}")

print()
if all_pass:
    print("All 7 verification checks PASSED.")
else:
    print("SOME CHECKS FAILED - see above.")
    sys.exit(1)
