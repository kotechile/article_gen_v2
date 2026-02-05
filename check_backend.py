import sys
import os

# Change WD to backend dir so 'import src' works
backend_dir = os.path.join(os.getcwd(), "topic_analysis", "backend")
root_dir = os.getcwd()
os.chdir(backend_dir)
sys.path.insert(0, backend_dir)
# Remove root dir from sys.path to avoid picking up root 'src'
if root_dir in sys.path:
    sys.path.remove(root_dir)

try:
    print(f"CWD: {os.getcwd()}")
    print(f"sys.path: {sys.path}")
    import src
    print(f"src path: {getattr(src, '__path__', 'No path')}")
    print(f"src file: {getattr(src, '__file__', 'No file')}")
    
    print("Attempting to import main...")
    from main import app
    print("Successfully imported app from main.")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    # Print traceback
    import traceback
    traceback.print_exc()
    sys.exit(1)
