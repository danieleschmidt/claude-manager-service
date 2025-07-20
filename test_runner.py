#!/usr/bin/env python3
"""
Test runner script for the Claude Manager Service
"""
import sys
import subprocess
import os

def run_tests():
    """Run the test suite with coverage reporting"""
    
    # Add src directory to Python path so tests can import modules
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/',
        '--verbose',
        '--cov=src',
        '--cov-report=html',
        '--cov-report=term-missing',
        '--cov-fail-under=80'
    ]
    
    print("Running test suite...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=os.path.dirname(__file__))
        print("\n✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Please install test dependencies:")
        print("pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)