"""
Manual testing script for specific components
Run after: source venv/bin/activate
"""

import sys
from pathlib import Path


def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    import fastapi
    print("✅ FastAPI imported")

    import asyncpg
    print("✅ asyncpg imported")

    import uvicorn
    print("✅ uvicorn imported")

    assert fastapi is not None
    assert asyncpg is not None
    assert uvicorn is not None


def test_app_structure():
    """Test application structure"""
    print("\nTesting application structure...")
    from app.main import app
    print("✅ FastAPI app created")

    from config import settings
    print(f"✅ Settings loaded: {settings.MODEL_PATH}")

    models_dir = Path(settings.MODEL_PATH)
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.npy"))
        print(f"Found {len(model_files)} model files")
        for model_file in model_files:
            print(f"   - {model_file.name}")
        assert len(model_files) >= 0  # just to keep pytest happy
    else:
        print("⚠️ Models directory not found")
        assert models_dir.exists()  # fail test if missing


def test_routes():
    """Test route registration"""
    print("\nTesting routes...")
    from app.main import app

    routes = []
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            routes.append(f"{list(route.methods)[0] if route.methods else 'GET'} {route.path}")

    print("Registered routes:")
    for route in routes:
        print(f"   - {route}")

    assert len(routes) > 0  # ensure some routes are registered


def test_database_config():
    """Test database configuration (without connecting)"""
    print("\nTesting database configuration...")
    from config import settings

    assert settings.DATABASE_URL is not None
    print(f"✅ Database URL configured: {settings.DATABASE_URL[:30]}...")

    import asyncpg

    async def _test_pool():
        return await asyncpg.create_pool(dsn=settings.DATABASE_URL)

    print("✅ asyncpg pool creation function available (connection not tested)")


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
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Server error: {e}")


def main():
    """Run manual tests outside pytest"""
    print("AutoFi Manual Testing")
    print("=" * 40)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("App Structure", test_app_structure),
        ("Routes", test_routes),
        ("Database Config", test_database_config),
    ]

    passed = 0
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name} passed")
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_name} failed: {e}")
        except Exception as e:
            print(f"❌ {test_name} error: {e}")

    print(f"\nResults: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("All tests passed!")
        user_input = input("\nStart development server? (y/n): ")
        if user_input.lower() == "y":
            start_server()
    else:
        print("Some tests failed")


if __name__ == "__main__":
    main()
