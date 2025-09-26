#!/bin/bash

# Build the web showcase
echo "Building web showcase..."
wasm-pack build --target web --out-dir pkg --dev

# Copy the HTML file
cp index.html pkg/

echo "Web showcase built successfully!"
echo "Serve with: python3 -m http.server 8000 -d pkg"