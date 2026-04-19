// AudioWorklet processor — forwards raw Float32 PCM frames from the mic
// back to the main thread. Runs in AudioWorkletGlobalScope (isolated, no DOM).
//
// Chrome gives us ~128-sample chunks by default at the context's sampleRate
// (we create the context at 16kHz, so no resampling needed here).

class PCMProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel && channel.length) {
      // Copy because the underlying buffer is reused by the worklet runtime.
      this.port.postMessage(channel.slice());
    }
    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
