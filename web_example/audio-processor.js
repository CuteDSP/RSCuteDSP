// AudioWorklet Processor for CuteDSP real-time processing
// Enhanced with multiple effects for microphone processing

// Simple Biquad Filter Implementation - Optimized
class SimpleBiquad {
    constructor() {
        this.x1 = 0; this.x2 = 0;
        this.y1 = 0; this.y2 = 0;
        this.a0 = 1; this.a1 = 0; this.a2 = 0;
        this.b0 = 1; this.b1 = 0; this.b2 = 0;
        this.cutoff = 1000;
        this.resonance = 0.7;
        this.sampleRate = 44100;
        this.needsUpdate = true;
    }

    lowpass(cutoff, resonance, sampleRate) {
        if (this.cutoff !== cutoff || this.resonance !== resonance || this.sampleRate !== sampleRate) {
            this.cutoff = cutoff;
            this.resonance = resonance;
            this.sampleRate = sampleRate;
            this.needsUpdate = true;
        }
    }

    updateCoefficients() {
        if (!this.needsUpdate) return;

        const w0 = 2 * Math.PI * this.cutoff / this.sampleRate;
        const cosw0 = Math.cos(w0);
        const sinw0 = Math.sin(w0);
        const alpha = sinw0 / (2 * this.resonance);

        const b0 = (1 - cosw0) / 2;
        const b1 = 1 - cosw0;
        const b2 = (1 - cosw0) / 2;
        const a0 = 1 + alpha;
        const a1 = -2 * cosw0;
        const a2 = 1 - alpha;

        this.b0 = b0 / a0; this.b1 = b1 / a0; this.b2 = b2 / a0;
        this.a1 = a1 / a0; this.a2 = a2 / a0;
        this.needsUpdate = false;
    }

    process(input, output) {
        this.updateCoefficients();

        for (let i = 0; i < input.length; i++) {
            const x = input[i];
            const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 -
                     this.a1 * this.y1 - this.a2 * this.y2;

            output[i] = y;

            this.x2 = this.x1; this.x1 = x;
            this.y2 = this.y1; this.y1 = y;
        }
    }
}

// Simple Delay Implementation - Optimized
class SimpleDelay {
    constructor(maxDelay, sampleRate) {
        this.maxDelay = maxDelay;
        this.sampleRate = sampleRate;
        this.buffer = new Float32Array(Math.ceil(maxDelay * sampleRate));
        this.writeIndex = 0;
        this.delaySamples = Math.floor(0.3 * sampleRate);
        this.feedback = 0.4;
    }

    set_delay_time(seconds) {
        this.delaySamples = Math.floor(seconds * this.sampleRate);
        if (this.delaySamples >= this.buffer.length) {
            this.delaySamples = this.buffer.length - 1;
        }
    }

    set_feedback(value) {
        this.feedback = value;
    }

    process_buffer(input, output) {
        for (let i = 0; i < input.length; i++) {
            const readIndex = (this.writeIndex - this.delaySamples + this.buffer.length) % this.buffer.length;
            const delayed = this.buffer[readIndex];

            output[i] = input[i] + delayed * this.feedback;

            this.buffer[this.writeIndex] = output[i];
            this.writeIndex = (this.writeIndex + 1) % this.buffer.length;
        }
    }
}

// Simple LFO Implementation - Optimized
class SimpleLFO {
    constructor() {
        this.phase = 0;
        this.frequency = 1;
        this.sampleRate = 44100;
        this.phaseIncrement = this.frequency / this.sampleRate;
    }

    set_frequency(freq, sampleRate) {
        this.frequency = freq;
        this.sampleRate = sampleRate;
        this.phaseIncrement = this.frequency / this.sampleRate;
    }

    process() {
        const value = Math.sin(this.phase * 2 * Math.PI);
        this.phase += this.phaseIncrement;
        if (this.phase >= 1) this.phase -= 1;
        return value;
    }
}

// Simple Distortion Effect
class SimpleDistortion {
    constructor() {
        this.drive = 1.0;
        this.amount = 0.5;
    }

    set_drive(value) {
        this.drive = value;
    }

    set_amount(value) {
        this.amount = value;
    }

    process(input, output) {
        for (let i = 0; i < input.length; i++) {
            const x = input[i] * this.drive;
            // Soft clipping distortion
            const y = x / (1 + Math.abs(x));
            output[i] = y * this.amount + input[i] * (1 - this.amount);
        }
    }
}

// Simple Reverb Effect (Schroeder reverb)
class SimpleReverb {
    constructor(sampleRate) {
        this.sampleRate = sampleRate;
        this.delays = [
            new SimpleDelay(0.1, sampleRate), // 100ms
            new SimpleDelay(0.05, sampleRate), // 50ms
            new SimpleDelay(0.025, sampleRate), // 25ms
            new SimpleDelay(0.0125, sampleRate)  // 12.5ms
        ];

        this.allpassDelays = [
            new SimpleDelay(0.006, sampleRate), // 6ms
            new SimpleDelay(0.003, sampleRate)  // 3ms
        ];

        this.mix = 0.3;
        this.decay = 0.7;
    }

    set_mix(value) {
        this.mix = value;
    }

    set_decay(value) {
        this.decay = value;
        // Update feedback on all delays
        this.delays.forEach(delay => delay.set_feedback(this.decay));
        this.allpassDelays.forEach(delay => delay.set_feedback(this.decay));
    }

    process(input, output) {
        // Start with input
        for (let i = 0; i < input.length; i++) {
            output[i] = input[i];
        }

        // Apply parallel comb filters
        const temp1 = new Float32Array(input.length);
        const temp2 = new Float32Array(input.length);
        const temp3 = new Float32Array(input.length);
        const temp4 = new Float32Array(input.length);

        this.delays[0].process_buffer(output, temp1);
        this.delays[1].process_buffer(output, temp2);
        this.delays[2].process_buffer(output, temp3);
        this.delays[3].process_buffer(output, temp4);

        // Mix comb filters
        for (let i = 0; i < input.length; i++) {
            output[i] = (temp1[i] + temp2[i] + temp3[i] + temp4[i]) * 0.25;
        }

        // Apply all-pass filters for diffusion
        this.allpassDelays[0].process_buffer(output, temp1);
        this.allpassDelays[1].process_buffer(temp1, temp2);

        // Mix dry/wet
        for (let i = 0; i < input.length; i++) {
            output[i] = input[i] * (1 - this.mix) + temp2[i] * this.mix;
        }
    }
}

// Simple Chorus Effect
class SimpleChorus {
    constructor(sampleRate) {
        this.sampleRate = sampleRate;
        this.delays = [
            new SimpleDelay(0.02, sampleRate), // 20ms base
            new SimpleDelay(0.025, sampleRate), // 25ms base
            new SimpleDelay(0.03, sampleRate)   // 30ms base
        ];
        this.lfos = [
            new SimpleLFO(),
            new SimpleLFO(),
            new SimpleLFO()
        ];

        // Set different LFO rates for chorus effect
        this.lfos[0].set_frequency(0.3, sampleRate);
        this.lfos[1].set_frequency(0.25, sampleRate);
        this.lfos[2].set_frequency(0.35, sampleRate);

        this.mix = 0.4;
        this.depth = 0.005; // 5ms modulation depth
    }

    set_mix(value) {
        this.mix = value;
    }

    set_depth(value) {
        this.depth = value;
    }

    process(input, output) {
        const temp1 = new Float32Array(input.length);
        const temp2 = new Float32Array(input.length);
        const temp3 = new Float32Array(input.length);

        // Process each voice with modulated delay
        for (let voice = 0; voice < 3; voice++) {
            const lfoValue = this.lfos[voice].process();
            const modulatedDelay = this.delays[voice].delaySamples + (lfoValue * this.depth * this.sampleRate);

            // Temporarily adjust delay time
            const originalDelay = this.delays[voice].delaySamples;
            this.delays[voice].delaySamples = Math.max(1, Math.floor(modulatedDelay));

            if (voice === 0) {
                this.delays[voice].process_buffer(input, temp1);
            } else if (voice === 1) {
                this.delays[voice].process_buffer(input, temp2);
            } else {
                this.delays[voice].process_buffer(input, temp3);
            }

            // Restore original delay
            this.delays[voice].delaySamples = originalDelay;
        }

        // Mix voices
        for (let i = 0; i < input.length; i++) {
            const chorus = (temp1[i] + temp2[i] + temp3[i]) / 3;
            output[i] = input[i] * (1 - this.mix) + chorus * this.mix;
        }
    }
}

// Simple Phaser Effect
class SimplePhaser {
    constructor(sampleRate) {
        this.sampleRate = sampleRate;
        this.allpassFilters = [];
        
        // Create 4 all-pass filters for the phaser
        for (let i = 0; i < 4; i++) {
            this.allpassFilters.push({
                delay: Math.floor((0.005 + i * 0.002) * sampleRate), // 5ms to 11ms delays
                feedback: 0.7,
                buffer: new Float32Array(Math.ceil(0.015 * sampleRate)), // 15ms max
                writeIndex: 0
            });
        }
        
        this.lfo = new SimpleLFO();
        this.lfo.set_frequency(0.5, sampleRate); // 0.5 Hz modulation
        
        this.mix = 0.5;
        this.depth = 0.5;
    }

    set_mix(value) {
        this.mix = value;
    }

    set_depth(value) {
        this.depth = value;
    }

    process(input, output) {
        // Start with input
        for (let i = 0; i < input.length; i++) {
            output[i] = input[i];
        }

        // Get LFO value for modulation
        const lfoValue = this.lfo.process();
        
        // Process through all-pass filters with modulated feedback
        for (let filter of this.allpassFilters) {
            const modulatedFeedback = filter.feedback * (1 + lfoValue * this.depth);
            
            for (let i = 0; i < input.length; i++) {
                const readIndex = (filter.writeIndex - filter.delay + filter.buffer.length) % filter.buffer.length;
                const delayed = filter.buffer[readIndex];
                
                const filtered = output[i] + delayed * modulatedFeedback;
                filter.buffer[filter.writeIndex] = filtered;
                filter.writeIndex = (filter.writeIndex + 1) % filter.buffer.length;
                
                output[i] = filtered * -modulatedFeedback + delayed;
            }
        }

        // Mix dry/wet
        for (let i = 0; i < input.length; i++) {
            output[i] = input[i] * (1 - this.mix) + output[i] * this.mix;
        }
    }
}

class CuteDSPProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.filter = null;
        this.delay = null;
        this.lfo = null;
        this.distortion = null;
        this.reverb = null;
        this.chorus = null;
        this.phaser = null;
        this.sampleRate = sampleRate;
        this.isInitialized = false;

        // Effect enable flags
        this.effectsEnabled = {
            filter: true,
            delay: true,
            distortion: false,
            reverb: false,
            chorus: false,
            phaser: false
        };

        // Pre-allocate buffers to avoid allocations during processing
        this.tempBuffer1 = new Float32Array(128);
        this.tempBuffer2 = new Float32Array(128);
        this.tempBuffer3 = new Float32Array(128);

        // Listen for messages from main thread
        this.port.onmessage = (event) => {
            const { type, data } = event.data;

            switch (type) {
                case 'loadWasm':
                    console.log('WebAssembly buffer received (using JS fallback)');
                    break;
                case 'init':
                    this.initializeDSP();
                    break;
                case 'updateFilter':
                    if (this.filter) {
                        this.filter.lowpass(data.cutoff, data.resonance, this.sampleRate);
                    }
                    break;
                case 'updateDelay':
                    if (this.delay) {
                        this.delay.set_delay_time(data.delayTime);
                        this.delay.set_feedback(data.feedback);
                    }
                    break;
                case 'updateLFO':
                    if (this.lfo) {
                        this.lfo.set_frequency(data.rate, this.sampleRate);
                    }
                    break;
                case 'updateDistortion':
                    if (this.distortion) {
                        this.distortion.set_drive(data.drive);
                        this.distortion.set_amount(data.amount);
                    }
                    break;
                case 'updateReverb':
                    if (this.reverb) {
                        this.reverb.set_mix(data.mix);
                        this.reverb.set_decay(data.decay);
                    }
                    break;
                case 'updateChorus':
                    if (this.chorus) {
                        this.chorus.set_mix(data.mix);
                        this.chorus.set_depth(data.depth);
                    }
                    break;
                case 'updatePhaser':
                    if (this.phaser) {
                        this.phaser.set_mix(data.mix);
                        this.phaser.set_depth(data.depth);
                    }
                    break;
                case 'toggleEffect':
                    if (this.effectsEnabled.hasOwnProperty(data.effect)) {
                        this.effectsEnabled[data.effect] = data.enabled;
                        console.log(`Effect ${data.effect} ${data.enabled ? 'enabled' : 'disabled'}`);
                    }
                    break;
            }
        };
    }

    async initializeDSP() {
        try {
            // Create JavaScript-based DSP components
            this.filter = new SimpleBiquad();
            this.filter.lowpass(1000.0, 0.7, this.sampleRate);

            this.delay = new SimpleDelay(1.0, this.sampleRate);
            this.delay.set_delay_time(0.3);
            this.delay.set_feedback(0.4);

            this.lfo = new SimpleLFO();
            this.lfo.set_frequency(1.0, this.sampleRate);

            this.distortion = new SimpleDistortion();
            this.distortion.set_drive(2.0);
            this.distortion.set_amount(0.3);

            this.reverb = new SimpleReverb(this.sampleRate);
            this.reverb.set_mix(0.3);
            this.reverb.set_decay(0.6);

            this.chorus = new SimpleChorus(this.sampleRate);
            this.chorus.set_mix(0.4);
            this.chorus.set_depth(0.005);

            this.phaser = new SimplePhaser(this.sampleRate);
            this.phaser.set_mix(0.5);
            this.phaser.set_depth(0.5);

            this.isInitialized = true;
            this.port.postMessage({ type: 'ready' });

        } catch (error) {
            this.port.postMessage({ type: 'error', error: error.message });
        }
    }

    process(inputs, outputs, parameters) {
        if (!this.isInitialized) {
            // Pass through audio unchanged if not initialized
            const input = inputs[0];
            const output = outputs[0];
            if (input && output) {
                for (let channel = 0; channel < input.length; channel++) {
                    const inputChannel = input[channel];
                    const outputChannel = output[channel];
                    for (let i = 0; i < inputChannel.length; i++) {
                        outputChannel[i] = inputChannel[i];
                    }
                }
            }
            return true;
        }

        const input = inputs[0];
        const output = outputs[0];

        if (!input || !output) return true;

        // Ensure buffers are large enough
        const bufferSize = input[0].length;
        if (this.tempBuffer1.length < bufferSize) {
            this.tempBuffer1 = new Float32Array(bufferSize);
            this.tempBuffer2 = new Float32Array(bufferSize);
            this.tempBuffer3 = new Float32Array(bufferSize);
        }

        // Process each channel
        for (let channel = 0; channel < input.length; channel++) {
            const inputBuffer = input[channel];
            const outputBuffer = output[channel];

            // Apply DSP chain
            this.processAudio(inputBuffer, outputBuffer, bufferSize);
        }

        return true;
    }

    processAudio(input, output, bufferSize) {
        // Start with input signal
        for (let i = 0; i < bufferSize; i++) {
            output[i] = input[i];
        }

        // Apply distortion first (if enabled)
        if (this.effectsEnabled.distortion) {
            this.distortion.process(output, this.tempBuffer1);
            [output, this.tempBuffer1] = [this.tempBuffer1, output]; // Swap buffers
        }

        // Apply filter with LFO modulation (if enabled)
        if (this.effectsEnabled.filter) {
            // Calculate LFO-modulated cutoff
            const lfoValue = this.lfo.process();
            const baseCutoff = 1000;
            const lfoDepth = 0.5;
            // Use efficient linear interpolation instead of expensive pow
            const modulation = 1 + (lfoValue * lfoDepth);
            const modulatedCutoff = Math.max(100, Math.min(8000, baseCutoff * modulation));

            this.filter.lowpass(modulatedCutoff, 0.7, this.sampleRate);
            this.filter.process(output, this.tempBuffer1);
            [output, this.tempBuffer1] = [this.tempBuffer1, output]; // Swap buffers
        }

        // Apply chorus (if enabled)
        if (this.effectsEnabled.chorus) {
            this.chorus.process(output, this.tempBuffer1);
            [output, this.tempBuffer1] = [this.tempBuffer1, output]; // Swap buffers
        }

        // Apply delay (if enabled)
        if (this.effectsEnabled.delay) {
            this.delay.process_buffer(output, this.tempBuffer1);
            // Mix dry/wet for delay
            for (let i = 0; i < bufferSize; i++) {
                this.tempBuffer1[i] = output[i] * 0.7 + this.tempBuffer1[i] * 0.3;
            }
            [output, this.tempBuffer1] = [this.tempBuffer1, output]; // Swap buffers
        }

        // Apply reverb last (if enabled)
        if (this.effectsEnabled.reverb) {
            this.reverb.process(output, this.tempBuffer1);
            [output, this.tempBuffer1] = [this.tempBuffer1, output]; // Swap buffers
        }

        // Apply phaser (if enabled)
        if (this.effectsEnabled.phaser) {
            this.phaser.process(output, this.tempBuffer1);
            [output, this.tempBuffer1] = [this.tempBuffer1, output]; // Swap buffers
        }
    }
}

registerProcessor('cute-dsp-processor', CuteDSPProcessor);