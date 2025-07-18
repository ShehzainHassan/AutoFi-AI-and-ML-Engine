#!/bin/bash
# Local Development Setup Script for AutoFi Vehicle Recommendation System

set -e  # Exit on any error

echo "🚀 Setting up AutoFi Vehicle Recommendation System locally..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Please run this script from the project root."
    exit 1
fi

# Create virtual environment
print_status "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_status "Installing requirements..."
pip install -r requirements.txt

# Install additional testing dependencies
print_status "Installing testing dependencies..."
pip install -r requirements-dev.txt

# Run local tests
print_status "Running local tests..."
python test_local.py

# Provide next steps
echo ""
echo "🎉 Setup complete!"
echo ""
echo "To work with your project:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the app locally: python start.py"
echo "3. Or run with uvicorn: uvicorn app.main:app --reload"
echo "4. Deactivate when done: deactivate"
echo ""
echo "To test manually:"
echo "- Health check: curl http://localhost:8000/"
echo "- API docs: http://localhost:8000/docs"
echo ""
echo "Ready for Railway deployment! 🚄" 