from src.brain import Brain
import sys

def debug_brain():
    print("Initializing Brain...")
    brain = Brain()
    
    # Test case that requires writing code
    test_input = "Write a python script in test_hello.py that prints hello world and calculates 2+2"
    
    print(f"\nProcessing command: '{test_input}'")
    try:
        response = brain.process_command(test_input)
        print("\n--- FINAL RESPONSE ---")
        print(response)
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_brain()
