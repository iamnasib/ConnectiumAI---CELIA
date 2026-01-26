from PIL import Image
import os

# Convert your images to raw bytes once locally
for i in range(1, 12):
    img = Image.open(f"assets/ConnectiumAI_visualizer_{i}.png").convert("RGB")
    with open(f"assets/ConnectiumAI_visualizer_{i}.bin", "wb") as f:
        f.write(img.tobytes())

# for i in range(1,12):
#             full_path = os.path.join(script_dir, f"assets/ConnectiumAI_visualizer_{i}.png")
           
#             with Image.open(full_path) as img:
                
#                 # Ensure RGB format
#                 if img.mode != 'RGB':
#                     img = img.convert('RGB')
                
#                 img_copy = img.copy()
#                 raw_img = OutputImageRawFrame(
#                     image=img_copy.tobytes(),
#                     size=img_copy.size,
#                     format='RGB'
#                 )
#                 sprites.append(raw_img)