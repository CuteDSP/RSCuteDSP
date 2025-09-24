#!/bin/bash

# CuteDSP Web Example Server
# Simple script to serve the WebAssembly demo

echo "🎵 CuteDSP WebAssembly Demo"
echo "============================"
echo ""
echo "Starting local server..."
echo "Open your browser to: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
python3 -m http.server 8080