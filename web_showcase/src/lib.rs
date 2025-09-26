use sycamore::prelude::*;
use wasm_bindgen::prelude::*;
use web_sys::{AudioContext, AudioBuffer, AudioBufferSourceNode, GainNode, AnalyserNode, HtmlInputElement};
use cute_dsp::filters::{Biquad, BiquadDesign};
use cute_dsp::delay::{Delay, InterpolatorLinear};
use cute_dsp::envelopes::CubicLfo;
use cute_dsp::curves::Reciprocal;
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Clone)]
struct AudioState {
    context: Option<AudioContext>,
    analyser: Option<AnalyserNode>,
    gain: Option<GainNode>,
    is_playing: bool,
}

#[component]
fn App<G: Html>(cx: Scope) -> View<G> {
    let audio_state = create_signal(cx, AudioState {
        context: None,
        analyser: None,
        gain: None,
        is_playing: false,
    });

    // Effect parameters
    let filter_freq = create_signal(cx, 1000.0f32);
    let filter_q = create_signal(cx, 0.7f32);
    let delay_time = create_signal(cx, 0.3f32);
    let delay_feedback = create_signal(cx, 0.4f32);
    let tremolo_rate = create_signal(cx, 5.0f32);
    let tremolo_depth = create_signal(cx, 0.5f32);
    let compressor_threshold = create_signal(cx, -20.0f32);
    let compressor_ratio = create_signal(cx, 4.0f32);

    // Effect enable states
    let filter_enabled = create_signal(cx, false);
    let delay_enabled = create_signal(cx, false);
    let tremolo_enabled = create_signal(cx, false);
    let compressor_enabled = create_signal(cx, false);

    let start_audio = move |_| {
        let mut state = audio_state.get().as_ref().clone();

        if state.context.is_none() {
            let context = AudioContext::new().unwrap();
            let analyser = context.create_analyser().unwrap();
            let gain = context.create_gain().unwrap();

            analyser.connect_with_audio_node(&gain).unwrap();
            gain.connect_with_audio_node(&context.destination()).unwrap();

            state.context = Some(context);
            state.analyser = Some(analyser);
            state.gain = Some(gain);
        }

        if let Some(context) = &state.context {
            if !state.is_playing {
                // Create a simple test tone
                let buffer = context.create_buffer(1, context.sample_rate() as u32 * 2, context.sample_rate()).unwrap();
                let mut channel_data = buffer.get_channel_data(0).unwrap();

                // Generate a test signal (sine wave sweep)
                for i in 0..channel_data.len() {
                    let t = i as f32 / context.sample_rate() as f32;
                    let freq = 220.0 + 880.0 * (t / 2.0).min(1.0);
                    channel_data[i] = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.1;
                }

                let source = context.create_buffer_source().unwrap();
                source.set_buffer(Some(&buffer));
                source.connect_with_audio_node(state.analyser.as_ref().unwrap()).unwrap();
                source.start().unwrap();

                state.is_playing = true;
            }
        }

        audio_state.set(state);
    };

    let stop_audio = move |_| {
        let mut state = audio_state.get().as_ref().clone();
        if let Some(context) = &state.context {
            context.close().unwrap();
            state.context = None;
            state.analyser = None;
            state.gain = None;
            state.is_playing = false;
            audio_state.set(state);
        }
    };

    view! { cx,
        div(class="container") {
            h1 { "CuteDSP Web Audio Effects" }

            div(class="controls") {
                button(on:click=start_audio, disabled=audio_state.get().is_playing) { "Start Audio" }
                button(on:click=stop_audio, disabled=!audio_state.get().is_playing) { "Stop Audio" }
            }

            div(class="effects") {
                // Filter Effect
                div(class="effect") {
                    h3 { "Filter" }
                    label {
                        input(type="checkbox", bind:checked=filter_enabled)
                        " Enable"
                    }
                    div(class="param") {
                        label { "Frequency: " (filter_freq.get()) " Hz" }
                        input(
                            type="range",
                            min="100",
                            max="8000",
                            step="10",
                            value=(filter_freq.get().to_string()),
                            on:input=move |e: web_sys::Event| {
                                if let Ok(val) = e.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value().parse::<f32>() {
                                    filter_freq.set(val);
                                }
                            }
                        )
                    }
                    div(class="param") {
                        label { "Q: " (filter_q.get()) }
                        input(
                            type="range",
                            min="0.1",
                            max="5.0",
                            step="0.1",
                            value=(filter_q.get().to_string()),
                            on:input=move |e: web_sys::Event| {
                                if let Ok(val) = e.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value().parse::<f32>() {
                                    filter_q.set(val);
                                }
                            }
                        )
                    }
                }

                // Delay Effect
                div(class="effect") {
                    h3 { "Delay" }
                    label {
                        input(type="checkbox", bind:checked=delay_enabled)
                        " Enable"
                    }
                    div(class="param") {
                        label { "Time: " (delay_time.get()) " s" }
                        input(
                            type="range",
                            min="0.1",
                            max="1.0",
                            step="0.01",
                            value=(delay_time.get().to_string()),
                            on:input=move |e: web_sys::Event| {
                                if let Ok(val) = e.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value().parse::<f32>() {
                                    delay_time.set(val);
                                }
                            }
                        )
                    }
                    div(class="param") {
                        label { "Feedback: " (delay_feedback.get()) }
                        input(
                            type="range",
                            min="0.0",
                            max="0.9",
                            step="0.01",
                            value=(delay_feedback.get().to_string()),
                            on:input=move |e: web_sys::Event| {
                                if let Ok(val) = e.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value().parse::<f32>() {
                                    delay_feedback.set(val);
                                }
                            }
                        )
                    }
                }

                // Tremolo Effect
                div(class="effect") {
                    h3 { "Tremolo" }
                    label {
                        input(type="checkbox", bind:checked=tremolo_enabled)
                        " Enable"
                    }
                    div(class="param") {
                        label { "Rate: " (tremolo_rate.get()) " Hz" }
                        input(
                            type="range",
                            min="0.1",
                            max="20.0",
                            step="0.1",
                            value=(tremolo_rate.get().to_string()),
                            on:input=move |e: web_sys::Event| {
                                if let Ok(val) = e.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value().parse::<f32>() {
                                    tremolo_rate.set(val);
                                }
                            }
                        )
                    }
                    div(class="param") {
                        label { "Depth: " (tremolo_depth.get()) }
                        input(
                            type="range",
                            min="0.0",
                            max="1.0",
                            step="0.01",
                            value=(tremolo_depth.get().to_string()),
                            on:input=move |e: web_sys::Event| {
                                if let Ok(val) = e.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value().parse::<f32>() {
                                    tremolo_depth.set(val);
                                }
                            }
                        )
                    }
                }

                // Compressor Effect
                div(class="effect") {
                    h3 { "Compressor" }
                    label {
                        input(type="checkbox", bind:checked=compressor_enabled)
                        " Enable"
                    }
                    div(class="param") {
                        label { "Threshold: " (compressor_threshold.get()) " dB" }
                        input(
                            type="range",
                            min="-60",
                            max="0",
                            step="1",
                            value=(compressor_threshold.get().to_string()),
                            on:input=move |e: web_sys::Event| {
                                if let Ok(val) = e.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value().parse::<f32>() {
                                    compressor_threshold.set(val);
                                }
                            }
                        )
                    }
                    div(class="param") {
                        label { "Ratio: " (compressor_ratio.get()) ":1" }
                        input(
                            type="range",
                            min="1",
                            max="20",
                            step="0.5",
                            value=(compressor_ratio.get().to_string()),
                            on:input=move |e: web_sys::Event| {
                                if let Ok(val) = e.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value().parse::<f32>() {
                                    compressor_ratio.set(val);
                                }
                            }
                        )
                    }
                }
            }

            div(class="visualizer") {
                canvas(id="spectrum", width="800", height="300")
            }
        }

        style { r#"
            .container {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
            }
            .controls {
                margin-bottom: 20px;
            }
            .controls button {
                margin-right: 10px;
                padding: 10px 20px;
                font-size: 16px;
            }
            .effects {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .effect {
                border: 1px solid #ccc;
                padding: 15px;
                border-radius: 8px;
            }
            .effect h3 {
                margin-top: 0;
                color: #333;
            }
            .param {
                margin: 10px 0;
            }
            .param label {
                display: block;
                margin-bottom: 5px;
            }
            .param input {
                width: 100%;
            }
            .visualizer {
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 10px;
            }
        "# }
    }
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn run_app() {
    console_error_panic_hook::set_once();
    sycamore::render_to(|cx| view! { cx, App {} }, &web_sys::window().unwrap().document().unwrap().get_element_by_id("app").unwrap());
}