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


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def test_check_requirements():
    """Check if all required files exist"""
    print_header("CHECKING REQUIREMENTS")

    required_files = [
        "requirements.txt",
        "app/main.py",
        "config/",
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

    assert not missing_files, f"Missing files: {missing_files}"
    print("\n✅ All required files found!")


def test_install_dependencies():
    """Install Python dependencies"""
    print_header("INSTALLING DEPENDENCIES")

    print("Installing requirements...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Installation failed:\n{result.stderr}")
    assert result.returncode == 0, "Dependency installation failed"
    print("✅ Dependencies installed successfully!")


def test_imports():
    """Test if all imports work"""
    print_header("TESTING IMPORTS")

    print("Testing FastAPI import...")
    import fastapi
    assert fastapi is not None
    print("✅ FastAPI imported successfully")

    print("Testing psycopg2 import...")
    import psycopg2
    assert psycopg2 is not None
    print("✅ psycopg2 imported successfully")

    print("Testing application imports...")
    from app.main import app
    assert app is not None
    print("✅ Main app imported successfully")

    from config import settings
    assert settings is not None
    print("✅ Config imported successfully")

    print("✅ All imports working!")


def test_app_creation():
    """Test if the FastAPI app can be created"""
    print_header("TESTING APP CREATION")

    from app.main import app
    assert app is not None
    print("✅ FastAPI app created successfully")

    routes = [route.path for route in app.routes]
    expected_routes = ["/", "/api/recommendations/user/{user_id}"]

    for route in expected_routes:
        found = any(route in r for r in routes)
        print(f"{'✅' if found else '⚠️'} Route check: {route}")
        assert found, f"Route not found: {route}"


def test_without_database():
    """Test components that don't require database"""
    print_header("TESTING WITHOUT DATABASE")

    from config import settings
    assert settings.MODEL_PATH is not None
    print(f"✅ Settings loaded: {settings.MODEL_PATH}")

    print("✅ Configuration test passed")


def test_server_startup():
    """Test if the server can start"""
    print_header("TESTING SERVER STARTUP")

    print("Starting server test...")
    print("Note: DB errors expected if PostgreSQL is not running")

    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "127.0.0.1",
            "--port", "8001",
            "--log-level", "info"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        time.sleep(10)

        try:
            response = requests.get("http://127.0.0.1:8001/", timeout=5)
            assert response.status_code == 200, f"Unexpected status {response.status_code}"
            print("✅ Server started successfully")
            print(f"✅ Health check: {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Server connection failed: {e}")
            # still assert so pytest fails
            assert False, f"Server not responding: {e}"
    finally:
        process.terminate()
        process.wait()
        stdout, stderr = process.communicate()
        print("🔍 Server stdout:\n", stdout)
        print("🔍 Server stderr:\n", stderr)



# -----------------
# Manual runner
# -----------------

def main():
    """Run all tests manually outside pytest"""
    print_header("AUTOFI LOCAL TESTING SUITE")
    print("Testing your application before Railway deployment...")

    tests = [
        ("Requirements Check", test_check_requirements),
        ("Dependencies Installation", test_install_dependencies),
        ("Import Tests", test_imports),
        ("App Creation Test", test_app_creation),
        ("Configuration Test", test_without_database),
        ("Server Startup Test", test_server_startup),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name} passed")
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_name} failed: {e}")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")

    print_header("TEST RESULTS")
    print(f"Passed: {passed}/{total} tests")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Your app is ready for Railway deployment!")
    else:
        print("❌ Some tests failed. Fix issues before deploying.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
