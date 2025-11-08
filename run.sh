#!/bin/bash

# Smart Visual Commerce Platform - Quick Start Script

echo "🛍️ Smart Visual Commerce Platform"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "✓ Dependencies installed"
echo ""

# Menu
echo "Select an option:"
echo "1) Run Streamlit Web App"
echo "2) Run FastAPI Server"
echo "3) Open Jupyter Notebook"
echo "4) Run All (API + Streamlit)"
echo "5) Exit"
echo ""

read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo "🚀 Starting Streamlit app..."
        streamlit run app/streamlit_app.py
        ;;
    2)
        echo "🚀 Starting FastAPI server..."
        python src/api/main.py
        ;;
    3)
        echo "🚀 Starting Jupyter Notebook..."
        jupyter notebook notebooks/demo_complete_system.ipynb
        ;;
    4)
        echo "🚀 Starting all services..."
        echo "API will be at: http://localhost:8000"
        echo "Streamlit will be at: http://localhost:8501"
        python src/api/main.py &
        streamlit run app/streamlit_app.py
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac
