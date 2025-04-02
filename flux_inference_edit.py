from PIL import Image, ImageDraw
import torch
from pipeline_flux_fill_edit import FluxFillEditPipeline
from diffusers.utils import load_image  # if needed elsewhere

# Define canvas dimensions 
canvas_width = 512 # full image width
canvas_height = 512   # full image height

# Load the input image locally (update the path as needed)
input_image = Image.open("val/img/40000.png")

# Resize the input image to a square of size (canvas_height, canvas_height)
input_resized = input_image.resize((canvas_height, canvas_height))

# Create the mask programmatically: left half white, right half black
mask = Image.new("RGB", (canvas_width, canvas_height), "white")

# Setup and configure the FluxFill pipeline as before
pipe = FluxFillEditPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev").to("mps")
pipe.load_lora_weights("flux-edit_circles-lora/checkpoint-500/pytorch_lora_weights.safetensors")
pipe.fuse_lora(lora_scale=1.0)

# example prompt for the fill circle dataset
result_image = pipe(
    prompt="An image of a circle on the left half, on the right half the same circle filled with red color in front of a yellow background",
    image=input_resized,
    mask_image=mask,
    height=canvas_height,
    width=canvas_width,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
    mode="edit"
).images[0]


result_image.save("flux-fill-dev.png")
