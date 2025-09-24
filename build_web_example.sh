#!/bin/bash

# CuteDSP WebAssembly Build Script
# Builds and prepares the web example with fresh WASM binaries

set -e  # Exit on any error

echo "🎵 CuteDSP WebAssembly Builder"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    print_error "Cargo.toml not found. Please run this script from the project root."
    exit 1
fi

print_status "Checking Rust toolchain..."
if ! command -v rustc &> /dev/null; then
    print_error "Rust is not installed. Please install Rust from https://rustup.rs/"
    exit 1
fi

print_status "Checking wasm-pack..."
if ! command -v wasm-pack &> /dev/null; then
    print_error "wasm-pack is not installed. Install with: cargo install wasm-pack"
    exit 1
fi

print_status "Building WebAssembly package..."
cd "$(dirname "$0")"  # Go to script directory (project root)

# Clean previous build
if [ -d "web_example/pkg" ]; then
    print_status "Cleaning previous build..."
    rm -rf web_example/pkg
fi

# Build WASM with optimizations
print_status "Compiling Rust to WebAssembly..."
wasm-pack build --target no-modules --out-dir web_example/pkg --features wasm

if [ $? -ne 0 ]; then
    print_error "WASM build failed!"
    exit 1
fi

print_success "WebAssembly build completed!"

# Verify the build
if [ ! -f "web_example/pkg/cute_dsp_bg.wasm" ]; then
    print_error "WASM file not found after build!"
    exit 1
fi

WASM_SIZE=$(stat -c%s "web_example/pkg/cute_dsp_bg.wasm" 2>/dev/null || stat -f%z "web_example/pkg/cute_dsp_bg.wasm" 2>/dev/null || echo "unknown")
print_success "WASM binary size: ${WASM_SIZE} bytes"

# Copy additional assets if needed
print_status "Preparing web assets..."

# Make sure the serve script is executable
chmod +x web_example/serve.sh

print_success "Build completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "  cd web_example"
echo "  ./serve.sh"
echo "  Open http://localhost:8080 in your browser"
echo ""
echo "📁 Build artifacts:"
echo "  • web_example/pkg/cute_dsp.js"
echo "  • web_example/pkg/cute_dsp_bg.wasm"
echo "  • web_example/index.html"
echo ""
print_success "Ready to serve! 🚀"