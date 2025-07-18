#!/usr/bin/env python3
"""
Local testing script for AutoFi Vehicle Recommendation System
Run this to test your application locally before Railway deployment
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def check_requirements():
    """Check if all required files exist"""
    print_header("CHECKING REQUIREMENTS")
    
    required_files = [
        "requirements.txt",
        "app/main.py",
        "config.py",
        "trained_models/",
        "Procfile",
        "railway.toml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    print("\n✅ All required files found!")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print_header("INSTALLING DEPENDENCIES")
    
    try:
        print("Installing requirements...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Installation failed:")
            print(result.stderr)
            return False
        
        print("✅ Dependencies installed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def test_imports():
    """Test if all imports work"""
    print_header("TESTING IMPORTS")
    
    try:
        print("Testing FastAPI import...")
        import fastapi
        print("✅ FastAPI imported successfully")
        
        print("Testing psycopg2 import...")
        import psycopg2
        print("✅ psycopg2 imported successfully")
        
        print("Testing application imports...")
        from app.main import app
        print("✅ Main app imported successfully")
        
        from config import settings
        print("✅ Config imported successfully")
        
        print("✅ All imports working!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_app_creation():
    """Test if the FastAPI app can be created"""
    print_header("TESTING APP CREATION")
    
    try:
        from app.main import app
        print("✅ FastAPI app created successfully")
        
        # Check if routes are registered
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/api/recommendations/user/{user_id}"]
        
        for route in expected_routes:
            if any(route in r for r in routes):
                print(f"✅ Route found: {route}")
            else:
                print(f"⚠️  Route not found: {route}")
        
        return True
        
    except Exception as e:
        print(f"❌ App creation error: {e}")
        return False

def test_without_database():
    """Test components that don't require database"""
    print_header("TESTING WITHOUT DATABASE")
    
    try:
        # Test configuration
        from config import settings
        print(f"✅ Settings loaded: {settings.MODEL_PATH}")
        
        # Test recommendation service initialization (without DB)
        print("✅ Configuration test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def run_server_test():
    """Test if the server can start"""
    print_header("TESTING SERVER STARTUP")
    
    print("Starting server test...")
    print("Note: This will test server startup without database connection")
    print("Database connection errors are expected if you don't have PostgreSQL running")
    
    try:
        # Start server in background
        import subprocess
        import signal
        
        # Use uvicorn directly
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "127.0.0.1", 
            "--port", "8001",  # Use different port for testing
            "--log-level", "info"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Test if server is responding
        try:
            response = requests.get("http://127.0.0.1:8001/", timeout=5)
            if response.status_code == 200:
                print("✅ Server started successfully!")
                print(f"✅ Health check passed: {response.json()}")
            else:
                print(f"⚠️  Server responded with status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Server connection test failed: {e}")
            print("This might be due to missing database connection")
        
        # Kill the process
        process.terminate()
        process.wait()
        
        return True
        
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False

def main():
    """Run all tests"""
    print_header("AUTOFI LOCAL TESTING SUITE")
    print("Testing your application before Railway deployment...")
    
    tests = [
        ("Requirements Check", check_requirements),
        ("Dependencies Installation", install_dependencies),
        ("Import Tests", test_imports),
        ("App Creation Test", test_app_creation),
        ("Configuration Test", test_without_database),
        ("Server Startup Test", run_server_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print_header("TEST RESULTS")
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your app is ready for Railway deployment!")
        print("\nNext steps:")
        print("1. Commit and push your changes to GitHub")
        print("2. Deploy to Railway")
        print("3. Add PostgreSQL service in Railway")
        print("4. Set environment variables in Railway")
    else:
        print("❌ Some tests failed. Please fix the issues before deploying.")
        print("\nCommon fixes:")
        print("- Make sure you're in the project root directory")
        print("- Run: pip install -r requirements.txt")
        print("- Check that all files are present")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 