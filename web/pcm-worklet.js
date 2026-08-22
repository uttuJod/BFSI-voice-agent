class PCM16Downsampler extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.outputChunkSize = 320;
    this.output = new Int16Array(this.outputChunkSize);
    this.outputIndex = 0;
    this.inputPosition = 0;
    this.nextOutputPosition = 0;
    this.step = sampleRate / this.targetRate;
  }

  emit(value) {
    const clipped = Math.max(-1, Math.min(1, value));
    this.output[this.outputIndex] = clipped < 0
      ? Math.round(clipped * 32768)
      : Math.round(clipped * 32767);
    this.outputIndex += 1;

    if (this.outputIndex >= this.outputChunkSize) {
      const packet = this.output;
      this.port.postMessage(packet.buffer, [packet.buffer]);
      this.output = new Int16Array(this.outputChunkSize);
      this.outputIndex = 0;
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }
    const mono = input[0];
    for (let i = 0; i < mono.length; i += 1) {
      const absolutePosition = this.inputPosition + i;
      while (absolutePosition >= this.nextOutputPosition) {
        this.emit(mono[i]);
        this.nextOutputPosition += this.step;
      }
    }
    this.inputPosition += mono.length;
    return true;
  }
}

registerProcessor("pcm16-downsampler", PCM16Downsampler);
