#!/bin/bash

# Build and serve the web showcase
./build.sh

echo "Starting server on http://localhost:8000"
python3 -m http.server 8000