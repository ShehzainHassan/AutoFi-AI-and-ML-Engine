#!/usr/bin/env python3
"""
Manual testing script for specific components
Run after: source venv/bin/activate
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    try:
        import fastapi
        print("✅ FastAPI imported")
        
        import psycopg2
        print("✅ psycopg2-binary imported")
        
        import uvicorn
        print("✅ uvicorn imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_app_structure():
    """Test application structure"""
    print("\nTesting application structure...")
    try:
        from app.main import app
        print("✅ FastAPI app created")
        
        from config import settings
        print(f"✅ Settings loaded: {settings.MODEL_PATH}")
        
        # Check trained models directory
        models_dir = Path(settings.MODEL_PATH)
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.npy"))
            print(f"✅ Found {len(model_files)} model files")
            for model_file in model_files:
                print(f"   - {model_file.name}")
        else:
            print("⚠️  Models directory not found")
        
        return True
    except Exception as e:
        print(f"❌ App structure error: {e}")
        return False

def test_routes():
    """Test route registration"""
    print("\nTesting routes...")
    try:
        from app.main import app
        
        routes = []
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                routes.append(f"{list(route.methods)[0] if route.methods else 'GET'} {route.path}")
        
        print("✅ Registered routes:")
        for route in routes:
            print(f"   - {route}")
        
        return True
    except Exception as e:
        print(f"❌ Routes error: {e}")
        return False

def test_database_config():
    """Test database configuration (without connecting)"""
    print("\nTesting database configuration...")
    try:
        from config import settings
        
        if settings.DATABASE_URL:
            print(f"✅ Database URL configured: {settings.DATABASE_URL[:30]}...")
        else:
            print("⚠️  DATABASE_URL not set")
        
        # Test database connection function (import only)
        from app.db import get_db_connection
        print("✅ Database connection function available")
        
        return True
    except Exception as e:
        print(f"❌ Database config error: {e}")
        return False

def start_server():
    """Start the development server"""
    print("\nStarting development server...")
    print("This will start the server on http://localhost:8000")
    print("Press Ctrl+C to stop")
    
    try:
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n✅ Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")

def main():
    """Run manual tests"""
    print("🧪 AutoFi Manual Testing")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("App Structure", test_app_structure),
        ("Routes", test_routes),
        ("Database Config", test_database_config),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
        
        user_input = input("\nStart development server? (y/n): ")
        if user_input.lower() == 'y':
            start_server()
    else:
        print("⚠️  Some tests failed")

if __name__ == "__main__":
    main() 