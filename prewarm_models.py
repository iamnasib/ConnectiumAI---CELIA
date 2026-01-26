"""
Pre-warm heavy models during Docker build.
This caches compiled ONNX models and imports so container startup is faster.
"""
import os
import sys

print("Pre-warming models for faster cold starts...")

# Force import all heavy pipecat modules (triggers compilation/caching)
from pipecat.transports.daily.transport import DailyTransport, DailyParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.openai.realtime.events import (
    SessionProperties,
    AudioConfiguration,
    AudioInput,
    InputAudioTranscription,
    InputAudioNoiseReduction,
    AudioOutput,
    SemanticTurnDetection,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import LLMRunFrame, EndFrame, SpriteFrame
from pipecat.processors.frame_processor import FrameProcessor

print("✓ Pipecat modules imported")

# Initialize Silero VAD once to cache the ONNX model
# This is the slowest part - it downloads/compiles the model
try:
    vad = SileroVADAnalyzer()
    print("✓ Silero VAD model cached")
except Exception as e:
    print(f"⚠ Could not pre-cache Silero VAD: {e}")

# Pre-compile Python files
import compileall
compileall.compile_dir('.', force=True, quiet=1)
print("✓ Python bytecode compiled")

print("Pre-warming complete!")
