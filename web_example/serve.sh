#!/bin/bash

# Simple HTTP server for the CuteDSP web example
# This script serves the web_example directory on localhost:8080

echo "🎵 CuteDSP Web Server"
echo "===================="

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found. Please install Python to run the web server."
    exit 1
fi

echo "🌐 Starting web server on http://localhost:8080"
echo "📁 Serving files from: $(pwd)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the HTTP server
$PYTHON_CMD -m http.server 8080

echo ""
echo "👋 Web server stopped."