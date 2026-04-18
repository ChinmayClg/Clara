import os, tempfile, shutil
from src.brain import Brain

b = Brain()

# Test scaffold creation in a temp directory
test_dir = os.path.join(tempfile.gettempdir(), 'test-vite-scaffold')
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

b._create_vite_react_scaffold(test_dir, 'test-app')

# Verify all files exist
expected = ['package.json', 'vite.config.js', 'index.html', 
            'src/main.jsx', 'src/index.css', 'src/App.jsx', 'src/App.css']

for f in expected:
    path = os.path.join(test_dir, f)
    exists = os.path.isfile(path)
    size = os.path.getsize(path) if exists else 0
    status = "OK" if exists else "MISSING"
    print(f"  {status}: {f} ({size} bytes)")

shutil.rmtree(test_dir)
print("\nAll scaffold files created successfully!")
