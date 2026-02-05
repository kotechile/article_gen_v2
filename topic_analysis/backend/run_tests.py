#!/usr/bin/env python3
"""
Test runner script for Supabase integration tests
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Supabase integration tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--contract", action="store_true", help="Run contract tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Default to running all tests if no specific type is specified
    if not any([args.unit, args.integration, args.contract, args.all]):
        args.all = True
    
    # Change to backend directory
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    print("🚀 Starting Supabase Integration Test Suite")
    print(f"Working directory: {os.getcwd()}")
    
    # Build pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        pytest_cmd.append("-v")
    
    if args.coverage:
        pytest_cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    if args.parallel:
        pytest_cmd.extend(["-n", str(args.parallel)])
    
    # Add test paths based on arguments
    test_paths = []
    
    if args.unit or args.all:
        test_paths.append("tests/unit/")
    
    if args.integration or args.all:
        test_paths.append("tests/integration/")
    
    if args.contract or args.all:
        test_paths.append("tests/contract/")
    
    if not test_paths:
        test_paths = ["tests/"]
    
    pytest_cmd.extend(test_paths)
    
    # Convert to string for subprocess
    command = " ".join(pytest_cmd)
    
    # Run tests
    success = run_command(command, "Supabase Integration Tests")
    
    if success:
        print("\n🎉 All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

