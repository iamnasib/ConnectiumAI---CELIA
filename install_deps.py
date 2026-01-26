from pipecat.audio.vad.silero import SileroVADAnalyzer

# Initialize to cache the ONNX model
vad = SileroVADAnalyzer()
print("Silero VAD model cached successfully")