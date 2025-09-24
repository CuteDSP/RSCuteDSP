# CuteDSP WebAssembly Example

A beautiful, interactive WebAssembly demo showcasing CuteDSP's high-performance digital signal processing capabilities.

## 🚀 Quick Start

### Option 1: Use the Build Script
```bash
cd /home/logic/RustroverProjects/RSCuteDSP
./build_and_serve.sh
```

### Option 2: Manual Setup
```bash
cd web_example
python3 -m http.server 8080
# Open http://localhost:8080 in your browser
```

## 🎵 Features

### FFT Analysis
- Real-time Fast Fourier Transform
- Adjustable FFT size (256-4096 points)
- Variable signal frequency generation
- Live spectrum visualization

### Audio Filtering
- Biquad filter implementation
- Multiple filter types: Low Pass, High Pass, Band Pass
- Adjustable cutoff frequency and Q factor
- Frequency response visualization

### Window Functions
- Multiple window types: Hann, Hamming, Blackman, Kaiser
- Real-time window visualization
- Spectral analysis improvements

### Performance Testing
- Benchmark DSP operations
- Measure processing speed
- Hardware concurrency detection

## 🛠️ Technical Details

- **Language**: Rust compiled to WebAssembly
- **Performance**: Native-speed DSP algorithms
- **Memory**: Efficient zero-copy operations
- **Compatibility**: Modern browsers with WebAssembly support
- **Size**: ~110KB WASM binary

## 📁 File Structure

```
web_example/
├── index.html          # Main demo page
├── pkg/               # WebAssembly package
│   ├── cute_dsp.js    # JavaScript bindings
│   ├── cute_dsp_bg.wasm  # Compiled WebAssembly
│   └── cute_dsp.d.ts  # TypeScript definitions
└── README.md          # This file
```

## 🎨 Demo Features

- **Responsive Design**: Works on desktop and mobile
- **Real-time Updates**: Interactive parameter controls
- **Canvas Visualization**: Live plotting of signals and spectra
- **Performance Monitoring**: Built-in benchmarking
- **Modern UI**: Gradient backgrounds and smooth animations

## 🔧 Browser Support

- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 16+

## 📊 Performance

The demo loads in under 2 seconds on modern hardware and provides real-time DSP processing at native speeds. All computation happens in the browser using WebAssembly.

## 🎯 Use Cases

- **Education**: Learn DSP concepts interactively
- **Development**: Test CuteDSP algorithms
- **Demonstration**: Showcase WebAssembly capabilities
- **Benchmarking**: Compare DSP performance across devices

## 🤝 Contributing

This example demonstrates the capabilities of the CuteDSP library. For library contributions, see the main project repository.

---

**Built with ❤️ using CuteDSP and WebAssembly**