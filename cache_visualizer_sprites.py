import pickle
import os
from pipecat.frames.frames import OutputImageRawFrame

sprites = []
script_dir = os.path.dirname(__file__)
full_path = os.path.join(script_dir, f"assets")

for i in range(1, 12):
    path = os.path.join(script_dir, f"assets/ConnectiumAI_visualizer_{i}.bin")
    with open(f"{path}", "rb") as f:
        raw_bytes = f.read()
        sprites.append(OutputImageRawFrame(image=raw_bytes, size=(1024, 576), format='RGB'))

# Save the whole list as one object
with open(f"{full_path}/sprites_cache.pkl", "wb") as f:
    pickle.dump(sprites, f)
