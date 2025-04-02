from PIL import Image, ImageDraw
import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image  # if needed elsewhere

# Define canvas dimensions 
canvas_width = 1024 
canvas_height = 512   

# Create a new composite image with a black background
canvas = Image.new("RGB", (canvas_width, canvas_height), "black")

# Load the input image locally (update the path as needed)
input_image = Image.open("val/img/40000.png")

# Resize the input image to a square of size (canvas_height, canvas_height)
input_resized = input_image.resize((canvas_height, canvas_height))

# Paste the resized input image onto the left side of the canvas
canvas.paste(input_resized, (0, 0))

# Create the mask programmatically: left half black, right half white
mask = Image.new("RGB", (canvas_width, canvas_height), "white")
draw_mask = ImageDraw.Draw(mask)
draw_mask.rectangle([0, 0, canvas_width // 2, canvas_height], fill="black")

# Setup and configure the FluxFill pipeline as before
pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev").to("mps")

pipe.load_lora_weights("flux-spacial_circles-lora/checkpoint-500/pytorch_lora_weights.safetensors")
pipe.fuse_lora(lora_scale=1.0)


# example prompt for the fill circle dataset
result_image = pipe(
    prompt="An image of a circle on the left half, on the right half the same circle filled with red color in front of a yellow background",
    image=canvas,
    mask_image=mask,
    height=canvas_height,
    width=canvas_width,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

result_image.save("flux-fill-dev.png")
