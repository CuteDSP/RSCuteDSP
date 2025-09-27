import init, { WasmBiquad } from './pkg/cute_dsp.js';

class FilterProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.filter = null;
    this.port.onmessage = async (event) => {
      if (event.data.type === 'init') {
        await init();
        this.filter = new WasmBiquad();
        this.filter.lowpass(event.data.freq, event.data.q, sampleRate);
      } else if (event.data.freq !== undefined) {
        if (this.filter) {
          this.filter.lowpass(event.data.freq, event.data.q, sampleRate);
        }
      }
    };
  }

  process(inputs, outputs) {
    const input = inputs[0][0];
    const output = outputs[0][0];
    output.set(input); // Temporarily pass through
    // if (this.filter) {
    //   this.filter.process(input, output);
    // } else {
    //   output.set(input);
    // }
    return true;
  }
}

registerProcessor('filter-processor', FilterProcessor);