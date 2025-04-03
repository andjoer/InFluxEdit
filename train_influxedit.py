#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# Modifications by Andreas Joerg
# Borrows from https://github.com/Sebastian-Zok/FLUX-Fill-LoRa-Training

import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image, ImageDraw
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset, BatchSampler, RandomSampler
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from pipeline_flux_fill_edit import FluxFillEditPipeline
from diffusers.utils import load_image
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
import safetensors.torch

from typing import Union

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

logger = get_logger(__name__)


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# Flux-Fill DreamBooth LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth LoRA weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with a custom [Flux diffusers trainer](https://github.com/andjoer/InFluxEdit).

Was LoRA for the text encoder enabled? {train_text_encoder}.

## Trigger words

You should use `{instance_prompt}` to trigger the image generation.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')
image = pipeline('{validation_prompt if validation_prompt else instance_prompt}').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "flux",
        "flux-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))

def resize_long_edge(image, target_size):
    # Get original width and height
    width, height = image.size
    if width >= height:
        new_width = target_size
        new_height = int(target_size * height / width)
    else:
        new_height = target_size
        new_width = int(target_size * width / height)
    return image.resize((new_width, new_height), resample=Image.BILINEAR)


def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)
    
    return mask, masked_image
    
def get_mask(im_shape, original_image_path, mask_data_path):
    # Extract only the file name from the original path
    _, filename = os.path.split(original_image_path)

    # Construct the mask path (adjust as needed)
    mask_path = os.path.join(mask_data_path, filename)
    # Load and ensure single-channel
    mask = Image.open(mask_path).convert("L")
    
    # Resize the mask to match your desired final image size if needed
    # Note: im_shape is typically (width, height)
    mask = mask.resize(im_shape, Image.NEAREST)

    return mask


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def log_validation(
    pipeline,
    args,
    accelerator,
    validation_prompt, # Specific prompt for this run
    val_image,         # PIL Image for input
    val_mask,          # PIL Image for mask (if not using split mask)
    mask_type,         # Optional: "left" or "right" if using split mask
    validation_index,  # Index of the validation pair
    is_final_validation=False,
):
    mask_info = f" with {mask_type} split mask" if mask_type else ""
    logger.info(
        f"Running validation for pair {validation_index}{mask_info}... \n Generating {args.num_validation_images} images with prompt:"
        f" {validation_prompt}."
    )
    pipeline.set_progress_bar_config(disable=False)

    # Determine the mask to use
    if mask_type:
        image_size = val_image.size
        mask_image = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask_image)
        if mask_type == "left":
            # Left half white.
            draw.rectangle([0, 0, image_size[0] // 2, image_size[1]], fill=255)
        elif mask_type == "right":
            # Right half white.
            draw.rectangle([image_size[0] // 2, 0, image_size[0], image_size[1]], fill=255)
        elif mask_type == "white":
            draw.rectangle([0, 0, image_size[0], image_size[1]], fill=255)
        else:
             raise ValueError(f"Invalid mask_type: {mask_type}")
    else:
        # Use the provided mask if not using split mask validation
        if val_mask is None:
             raise ValueError("val_mask must be provided if mask_type is not set.")
        mask_image = val_mask

    # Construct pipeline_args for this specific run
    pipeline_args = {"prompt": validation_prompt, "image": val_image, "mask_image": mask_image, "mode": args.mode}

    # run inference
    # Offset seed per pair *and* per mask type if applicable
    seed_offset = validation_index * 2 + (0 if mask_type != "right" else 1)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + seed_offset) if args.seed else None
    autocast_ctx = nullcontext()

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator, height=args.validation_height,width=args.validation_width).images[0] for _ in range(args.num_validation_images)]


    if  "wandb" in [tracker.name for tracker in accelerator.trackers]:
        phase_name = "test" if is_final_validation else "validation"
        log_key_suffix = f"_{mask_type}" if mask_type else ""
        log_key = f"{phase_name}_pair_{validation_index}{log_key_suffix}"
        caption_suffix = f" ({mask_type} mask)" if mask_type else ""
        wandb_images = [
            wandb.Image(image, caption=f"Pair {validation_index}, Sample {i}: {validation_prompt}")
            for i, image in enumerate(images)
        ]

    return {log_key: wandb_images}, images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    
    parser.add_argument(
        "--mask_data_dir",
        type=str,
        default=None,
        help=("A folder containing the mask data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt (or comma-separated list of prompts) that is used during validation.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help="A file path (or comma-separated list of paths) to image(s) used during validation.",
        # removed nargs="+"
    )
    parser.add_argument(
        "--validation_mask",
        type=str,
        default=None,
        help="A file path (or comma-separated list of paths) to mask(s) used during validation. Ignored if --use_split_mask is set.",
        # removed nargs="+"
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X steps. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--validation_width",
        type=int,
        default=1024,
        help=(
            "width of the validation images"
        ),
    )
    parser.add_argument(
        "--validation_height",
        type=int,
        default=1024,
        help=(
            "height of the validation images"
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=str,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma seperated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )

    parser.add_argument(
        "--size",
        type=str,
        default="512",
        help="Target size for images. Provide a single integer (e.g. 512) to simply resize, or a comma-separated list (e.g. '256,512,768') for random cropping to one of these sizes."
    )
    parser.add_argument(
        "--use_split_mask",
        action="store_true",
        help="If set, a synthetic mask is generated with either the left or right half white."
    )
    parser.add_argument(
        "--caption_data_dir",
        type=str,
        default=None,
        help="Folder containing prompt text files (with same filenames as images, but .txt extension)."
    )
    parser.add_argument(
        "--mask_column",
        type=str,
        default=None,
        help="(Optional) If using a dataset, the name of the mask column."
    )

    parser.add_argument(
        "--target_data_dir",
        type=str,
        default=None,
        help="Folder containing the target images"
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default=None,
        help="column name of the target images"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mask",
        help="mask or edit"
    )

    parser.add_argument(
        "--train_embedder",
        action="store_true",
        help="train the embedder"
    )
    parser.add_argument(
        "--mask_only_right",
        action="store_true",
        help="mask only the right half of the image"
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")
    
    if args.mask_data_dir is None and args.mask_column is None and not args.use_split_mask and args.mode == "mask":
        raise ValueError("Specify a --mask_data_dir or --mask_column")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args



class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance, target (optional), and class images with the prompts for fine-tuning the model.
    Loads PIL images and prompts. Resizing, mask generation/handling, and tensor conversion happen in collate_fn.
    """
    def __init__(
        self,
        instance_data_root,
        mask_data_root,   # currently in refactoring
        instance_prompt,
        class_prompt,
        class_data_root=None,
        class_num=None,
        repeats=1,
    ):
        global args # Access global args for mode, columns, etc.
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.repeats = repeats
        self.prompt_data_dir = args.caption_data_dir
        self.mode = args.mode # Store the mode

        # --- Data Source Setup ---
        self.dataset = args.dataset_name
        self.instance_data_root = args.instance_data_dir
        self.target_data_root = args.target_data_dir
        self.mask_data_dir = args.mask_data_dir # Initialize mask root
        self.instance_images_paths = []
        self.target_images_paths = {} # Only used in 'edit' mode with folders
        self.image_column = args.image_column
        self.target_column = args.target_column # Only used in 'edit' mode with dataset
        self.mask_column = args.mask_column # Used in 'mask' mode with dataset
        self.use_split_mask = args.use_split_mask
        self.mask_only_right = args.mask_only_right

        # --- Load image paths/dataset info ---
        self.mask_column = None
        if args.dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError("Please install the datasets library: pip install datasets")
            logger.info(f"Loading dataset {args.dataset_name}")
            self.dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir)["train"]
            self.num_samples = len(self.dataset)
            self.image_column = args.image_column if args.image_column is not None else self.dataset.column_names[0]
            # Check for mask column only if dataset is used
            self.mask_column = args.mask_column if args.mask_column in self.dataset.column_names else None
            logger.info(f"Using image column: {self.image_column}, mask column: {self.mask_column}")

            if args.caption_column is not None:
                if args.caption_column not in self.dataset.column_names:
                     raise ValueError(f"Caption column '{args.caption_column}' not found in dataset.")
                self.custom_instance_prompts = list(self.dataset[args.caption_column])
            else:
                self.custom_instance_prompts = None

            if args.mode == "edit" and args.target_column is not None:
                self.targets = list(self.dataset[args.target_column])

                
        else:
            # --- Load from instance_data_dir ---
            self.dataset = None
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.is_dir():
                raise ValueError(f"Instance data root is not a directory: {self.instance_data_root}")
            logger.info(f"Loading images from {self.instance_data_root}")
            # List files, filter common non-image types if needed, but _load_image handles errors
            self.instance_images_paths = sorted([p for p in self.instance_data_root.iterdir() if p.is_file()])
            self.num_samples = len(self.instance_images_paths)
            if self.num_samples == 0:
                 raise ValueError(f"No image files found in {self.instance_data_root}")
            logger.info(f"Found {self.num_samples} instance images.")
            self.custom_instance_prompts = None # No custom prompts when loading from dir unless caption_data_dir is used


        # --- Handle class images (load paths) ---
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            if not self.class_data_root.is_dir():
                 logger.warning(f"Class data root is not a directory: {self.class_data_root}")
                 self.class_data_root = None # Disable prior preservation if dir is invalid
                 self._length = self.num_samples * self.repeats
            else:
                 self.class_data_root.mkdir(parents=True, exist_ok=True)
                 self.class_images_paths = sorted([p for p in self.class_data_root.iterdir() if p.is_file()])
                 num_found = len(self.class_images_paths)
                 if class_num is not None:
                     self.num_class_images = min(num_found, class_num)
                 else:
                     self.num_class_images = num_found

                 if self.num_class_images > 0:
                     self._length = max(self.num_samples * self.repeats, self.num_class_images)
                     logger.info(f"Found {num_found} class images, using {self.num_class_images}.")
                 else:
                      logger.warning(f"No class images found in {self.class_data_root}. Disabling prior preservation for length calculation.")
                      self.class_data_root = None # Disable if no images found
                      self._length = self.num_samples * self.repeats
        else:
            self.class_data_root = None
            self._length = self.num_samples * self.repeats

        # NO image transforms defined here

    def __len__(self):
        # Ensure length reflects actual data available, especially if prior preservation disabled
        if self.class_data_root is not None and self.num_class_images > 0:
             return max(self.num_samples * self.repeats, self.num_class_images)
        else:
             return self.num_samples * self.repeats


    def _load_image(self, source) -> Union[Image.Image, None]:
        """Loads an image, converts to RGB, applies EXIF transpose."""
        try:
            if isinstance(source, (str, Path)):
                if not os.path.exists(source):
                     logger.error(f"Image file not found: {source}")
                     return None
                image = Image.open(source)
            elif isinstance(source, Image.Image):
                image = source
            else:
                logger.warning(f"Unexpected image source type: {type(source)}")
                return None

            image = exif_transpose(image) # Apply EXIF orientation correction

            if image.mode == "RGBA":
                image = image.convert("RGB")
            elif image.mode == "L":
                image = image.convert("RGB")
            elif image.mode != "RGB":
                logger.warning(f"Image mode is {image.mode} for {source}. Attempting conversion to RGB.")
                image = image.convert("RGB")

            return image
        except Exception as e:
            logger.error(f"Error loading/processing image {source}: {e}")
            return None

    def _get_mask(self, image_path_or_id, image_size) -> Union[Image.Image, None]:
        """Loads or generates mask as PIL Image (L mode). No resizing here."""
        if self.use_split_mask:
            mask = Image.new("L", image_size, 0)
            draw = ImageDraw.Draw(mask)
            half_width = image_size[0] // 2
            if random.random() < 0.5 and not self.mask_only_right:
                draw.rectangle([0, 0, half_width, image_size[1]], fill=255) # Left half
            else:
                draw.rectangle([half_width, 0, image_size[0], image_size[1]], fill=255) # Right half
            return mask

        if not self.mask_data_dir:
             logger.error("Mask data directory not provided, but needed (and not using split mask or dataset column).")
             # Return a default black mask, as finding one is impossible.
             return Image.new("L", image_size, 0)

        # Construct mask path based on image filename
        try:
            base = os.path.splitext(os.path.basename(str(image_path_or_id)))[0]
            potential_exts = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"] # Common mask extensions
            mask_path = None

            for ext in potential_exts:
                candidate = Path(self.mask_data_dir) / (base + ext)
                if candidate.is_file():
                    mask_path = candidate
                    break

            # Fallback: try exact name match if extensions failed (e.g., mask has same name as image)
            if mask_path is None:
                 candidate = Path(self.mask_data_dir) / os.path.basename(str(image_path_or_id))
                 if candidate.is_file():
                      mask_path = candidate

            if mask_path is None:
                logger.warning(f"Mask not found for image '{image_path_or_id}' in {self.mask_data_dir}. Using default black mask.")
                return Image.new("L", image_size, 0)

            mask = Image.open(mask_path).convert("L")
            return mask
        except Exception as e:
            logger.error(f"Error loading mask for {image_path_or_id} (path: {mask_path}): {e}. Using default black mask.")
            return Image.new("L", image_size, 0)

    def _get_target(self, image_path_or_id, image_size) -> Union[Image.Image, None]:
        """Loads or generates mask as PIL Image (L mode). No resizing here."""

        if not args.target_data_dir:
             logger.error("Target data directory not provided, but needed (and not using split mask or dataset column).")
             # Return a default black mask, as finding one is impossible.
             return Image.new("RGB", image_size, 0)

        # Construct mask path based on image filename
        try:
            base = os.path.splitext(os.path.basename(str(image_path_or_id)))[0]
            potential_exts = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"] # Common mask extensions
            target_path = None

            for ext in potential_exts:
                candidate = Path(args.target_data_dir) / (base + ext)
                if candidate.is_file():
                    target_path = candidate
                    break

            # Fallback: try exact name match if extensions failed (e.g., mask has same name as image)
            if target_path is None:
                 candidate = Path(args.target_data_dir) / os.path.basename(str(image_path_or_id))
                 if candidate.is_file():
                      target_path = candidate

            if target_path is None:
                logger.warning(f"Target not found for image '{image_path_or_id}' in {args.target_data_dir}. Using default black target.")
                return Image.new("RGB", image_size, 0)

            target = Image.open(target_path).convert("RGB")
            return target
        except Exception as e:
            logger.error(f"Error loading target for {image_path_or_id} (path: {target_path}): {e}. Using default black target.")
            return Image.new("L", image_size, 0)
        
    def _get_prompt(self, image_path_or_id, index):
        """Gets prompt: custom>caption_file>default instance prompt."""
        # 1. Check custom list from dataset (if applicable)
        if self.dataset is not None and self.custom_instance_prompts:
            # Use modulo index for repeats compatibility
            prompt = self.custom_instance_prompts[index % self.num_samples]
            if isinstance(prompt, str) and prompt.strip():
                return prompt.strip()

        # 2. Check caption data dir (if applicable)
        if self.prompt_data_dir is not None:
            try:
                base = os.path.splitext(os.path.basename(str(image_path_or_id)))[0]
                prompt_path = Path(self.prompt_data_dir) / (base + ".txt")
                if prompt_path.is_file():
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        prompt = f.read().strip()
                    if prompt:
                        return prompt
            except Exception as e:
                 logger.warning(f"Error reading prompt file {prompt_path}: {e}")

        # 3. Fallback to default instance prompt
        if not self.instance_prompt:
             logger.warning(f"No specific prompt found for {image_path_or_id} and instance_prompt is not set. Using empty prompt.")
             return ""
        return self.instance_prompt

    def __getitem__(self, index):
        """Loads PIL image, PIL mask, and prompt for the given index."""
        pil_image = None
        pil_mask = None
        image_path_id = None # Identifier for finding masks/prompts

        # Determine the effective index based on repeats and available data
        effective_instance_index = index % (self.num_samples * self.repeats)
        instance_sample_index = effective_instance_index % self.num_samples

        try:
            if self.dataset is not None:
                # --- Load from Dataset ---
                sample = self.dataset[instance_sample_index]
                pil_image = self._load_image(sample[self.image_column])

                if pil_image is None:
                     logger.warning(f"Skipping dataset index {instance_sample_index} due to image loading error.")
                     # Need robust way to handle skips, returning None might work with filtering in collate_fn
                     return None # Signal failure

                # Get mask: Check column first, then use _get_mask
                if self.mode == "edit":
                    pil_mask = Image.new("L", pil_image.size, 1)
                    pil_target = sample[self.target_column]
                elif self.use_split_mask:
                    pil_mask = self._get_mask(pil_image, pil_image.size)
                else:
                    mask_source = None
                    if self.mask_column and self.mask_column in sample:
                        mask_source = sample[self.mask_column]
                    if isinstance(mask_source, Image.Image):
                        pil_mask = mask_source.convert("L")

            else:
                # --- Load from instance_data_dir ---
                image_path = self.instance_images_paths[instance_sample_index]
                image_path_id = str(image_path)
                pil_image = self._load_image(image_path)

                if pil_image is None:
                     logger.warning(f"Skipping file index {instance_sample_index} ({image_path_id}) due to image loading error.")
                     return None # Signal failure

                # Get mask using the image path
                if self.mode == "edit":
                    pil_mask = Image.new("L", pil_image.size, 1)
                    pil_target = self._get_target(image_path, pil_image.size)
                else:
                    pil_mask = self._get_mask(image_path, pil_image.size)

            # --- Final Checks and Prompt ---
            if pil_mask is None: # Should have default, but double check
                 logger.error(f"Mask is None for {image_path_id}, skipping item.")
                 return None

            prompt = self._get_prompt(image_path_id, instance_sample_index)

            example = {
                "instance_pil_image": pil_image,
                "instance_pil_mask": pil_mask,
                "instance_prompt": prompt,
            }

            if self.mode == "edit":
                example["instance_pil_target"] = pil_target

            # --- Handle Class Images (Prior Preservation) ---
            if self.class_data_root is not None and self.num_class_images > 0:
                # Use index modulo num_class_images to cycle through class images
                class_index = index % self.num_class_images
                class_image_path = self.class_images_paths[class_index]
                class_pil_image = self._load_image(class_image_path)
                if class_pil_image:
                    example["class_pil_image"] = class_pil_image
                    example["class_prompt"] = self.class_prompt
                else:
                    logger.warning(f"Failed to load class image {class_image_path}")

            return example

        except Exception as e:
             logger.error(f"Unexpected error processing index {index} (instance index {instance_sample_index}, id: {image_path_id}): {e}", exc_info=True)
             return None


def collate_fn(examples, with_prior_preservation=False):
    global args # Access global args to get the resolution list

    # Filter out None items from dataset failures
    examples = [ex for ex in examples if ex is not None]
    if not examples:
        return {} # Return empty batch if all examples failed

    # --- Determine Target Resolution for the Batch ---
    size_list = args.resolution.split(',')
    if len(size_list) > 1: # Check if size_list exists and is not empty
        target_size = int(random.choice(size_list))
    else: # Check for single_size
        target_size = int(size_list[0])

    # --- Define Transforms dynamically based on chosen resolution ---
    image_transforms = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    mask_transforms = transforms.Compose([
         transforms.ToTensor(), # Converts to [0, 1] range
    ])

    # --- Process Batch Items ---
    instance_images = []
    instance_targets = []
    masks = []
    prompts = []
    class_images = [] # For prior preservation
    class_prompts = [] # For prior preservation

    for ex in examples:
        # Process Instance Image
        try:
            ex_image = resize_long_edge(ex["instance_pil_image"], target_size)
            instance_images.append(image_transforms(ex_image))
            if args.mode == "edit":
                ex_target = resize_long_edge(ex["instance_pil_target"], target_size)
                instance_targets.append(image_transforms(ex_target))
        except Exception as e:
            logger.error(f"Failed to transform instance image: {e}. Skipping this example in batch.")
            continue # Skip this example if transform fails

        # Process Mask
        try:
            ex_mask = resize_long_edge(ex["instance_pil_mask"], target_size)
            mask_tensor = mask_transforms(ex_mask)
            # Threshold mask AFTER resizing and ToTensor
            mask_tensor = (mask_tensor > 0.5).float() # Binary mask (0.0 or 1.0)
            masks.append(mask_tensor)
        except Exception as e:
            # If mask fails, we probably should skip the whole example
            logger.error(f"Failed to transform instance mask: {e}. Skipping this example in batch.")
            # Remove the already added image for this failed example
            if instance_images: instance_images.pop()
            continue

        # Get Prompt
        prompts.append(ex["instance_prompt"])

        # Process Class Image (if using prior preservation and available)
        if with_prior_preservation and "class_pil_image" in ex:
            try:
                 class_images.append(image_transforms(ex["class_pil_image"]))
                 class_prompts.append(ex["class_prompt"])
            except Exception as e:
                 logger.warning(f"Failed to transform class image: {e}. Omitting class data for this example.")
                 # Don't add prompt if image failed
                 pass # Continue with the instance data

    # If all items failed transformation, return empty
    if not instance_images:
        return {}

    # --- Stack Tensors ---
    try:
        pixel_values = torch.stack(instance_images)
        masks_stacked = torch.stack(masks)

        # Compute masked_images: image * (1 - mask) where mask=0 for region to keep,1 for region to mask out
        if args.mode == "edit":
            masked_images = pixel_values
        else:
            masked_images = pixel_values * (masks_stacked < 0.5).float() 

        batch = {
            "pixel_values": pixel_values,
            "prompts": prompts,
            "masks": masks_stacked,
            "masked_images": masked_images
        }

        if args.mode == "edit":
            batch["targets"] = torch.stack(instance_targets)

        # Add class images if processed
        if with_prior_preservation and class_images:
            # Ensure prompts match the number of successfully processed class images
            valid_class_prompts = class_prompts[:len(class_images)]
            if len(valid_class_prompts) != len(class_images):
                 logger.error("Mismatch between processed class images and prompts count.")
            else:
                 batch["class_images"] = torch.stack(class_images)
                 batch["class_prompts"] = valid_class_prompts

        return batch
    except Exception as e:
         logger.error(f"Error stacking tensors in collate_fn: {e}", exc_info=True)
         # Return empty dict or raise error depending on desired robustness
         return {}



class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def prepare_mask_latents(
        mask,
        masked_image_latents,
        batch_size,
        num_channels_latents,
        num_images_per_prompt,
        height,
        width,
        dtype,
        device,
        vae_scale_factor,
        vae_shift_factor
     ):
        """ Prepare mask latents """
        # 1. calculate the height and width of the latents
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

  

        masked_image_latents = (masked_image_latents - vae_shift_factor) * vae_scale_factor 
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        # 3. duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        batch_size = batch_size * num_images_per_prompt
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        # 4. pack the masked_image_latents
        # batch_size, num_channels_latents, height, width -> batch_size, height//2 * width//2 , num_channels_latents*4
        masked_image_latents = FluxFillEditPipeline._pack_latents(
            masked_image_latents,
            batch_size,
            num_channels_latents,
            height,
            width,
        )

        # 5.resize mask to latents shape we we concatenate the mask to the latents
        mask = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
        mask = mask.view(
            batch_size, height, vae_scale_factor, width, vae_scale_factor
        )  # batch_size, height, 8, width, 8
        mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
        mask = mask.reshape(
            batch_size, vae_scale_factor * vae_scale_factor, height, width
        )  # batch_size, 8*8, height, width

        # 6. pack the mask:
        # batch_size, 64, height, width -> batch_size, height//2 * width//2 , 64*2*2
        mask = FluxFillEditPipeline._pack_latents(
            mask,
            batch_size,
            vae_scale_factor * vae_scale_factor,
            height,
            width,
        )
        mask = mask.to(device=device, dtype=dtype)

        return mask, masked_image_latents

def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            has_supported_fp16_accelerator = torch.cuda.is_available() or torch.backends.mps.is_available()
            torch_dtype = torch.float16 if has_supported_fp16_accelerator else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = FluxFillEditPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                revision=args.revision,
                variant=args.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
 
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxFillEditPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                 if args.train_text_encoder: # Only expect to load if it was trained
                      text_encoder_one_ = model
                 # else: pass # Don't assign if not trained/loaded
            else:
                raise ValueError(f"unexpected load model: {model.__class__}")

        # Load LoRA weights
        x_embedder_path = os.path.join(input_dir, "flux_x_embedder.safetensors")

        if os.path.exists(x_embedder_path):
            lora_dir = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
        else: 
            lora_dir = input_dir
        lora_state_dict = FluxFillEditPipeline.lora_state_dict(lora_dir)

        # Load LoRA into Transformer
        if transformer_ is not None:
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
            }
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading transformer LoRA weights led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

            # Load the separately saved x_embedder weights
            if os.path.exists(x_embedder_path):
                try:
                    x_embedder_state_dict = safetensors.torch.load_file(x_embedder_path)
                    # Ensure requires_grad is True before loading, although load_state_dict usually preserves it
                    for param in transformer_.x_embedder.parameters():
                         param.requires_grad = True
                    transformer_.x_embedder.load_state_dict(x_embedder_state_dict)
                    logger.info(f"Successfully loaded x_embedder weights from {x_embedder_path}")
                except Exception as e:
                    logger.error(f"Failed to load x_embedder weights from {x_embedder_path}: {e}")
            else:
                logger.warning(f"x_embedder weights file not found at {x_embedder_path}. Skipping x_embedder loading.")


        # Load LoRA into Text Encoder (if applicable)
        if args.train_text_encoder and text_encoder_one_ is not None:
             # Check if text_encoder LoRA weights exist in the loaded dict
             if any(k.startswith("text_encoder.") for k in lora_state_dict):
                  _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)
                  logger.info("Loaded text_encoder LoRA weights.")
             else:
                  logger.warning("Attempting to load text_encoder LoRA, but no weights found in state_dict.")
        elif args.train_text_encoder and text_encoder_one_ is None:
             logger.warning("train_text_encoder is True, but text_encoder_one_ model instance was not found during loading hook.")


        # Make sure the trainable params are in float32. This includes LoRA and x_embedder.
        # This needs to happen *after* loading weights.
        models_to_cast = []
        if transformer_ is not None:
             models_to_cast.append(transformer_)
        if args.train_text_encoder and text_encoder_one_ is not None:
             models_to_cast.append(text_encoder_one_)

        if args.mixed_precision == "fp16":
            if models_to_cast:
                 # only upcast trainable parameters (LoRA + x_embedder) into fp32
                 cast_training_params(models_to_cast)
                 logger.info("Cast trainable parameters (LoRA and x_embedder) to fp32 for fp16 mixed precision training.")
        elif args.mixed_precision == "bf16":
             # For bf16, trainable params usually stay in bf16, but check specific needs
             logger.info("Using bf16 mixed precision. Trainable parameters remain in bf16.")
             # Optionally, you could still cast specific layers like x_embedder if needed:
             # cast_training_params(transformer_.x_embedder, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        if args.train_text_encoder:
            models.extend([text_encoder_one])
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    if args.train_text_encoder:
        # different learning rate for text encoder and unet
        text_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr, text_parameters_one_with_lr]
    else:
        params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        mask_data_root=args.mask_data_dir,
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        repeats=args.repeats,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts and args.instance_prompt is not None:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds, instance_text_ids = compute_text_embeddings(
            args.instance_prompt, text_encoders, tokenizers
        )

    # Handle class prompt for prior-preservation.
    if args.with_prior_preservation:
        if not args.train_text_encoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds, class_text_ids = compute_text_embeddings(
                args.class_prompt, text_encoders, tokenizers
            )

    # Clear the memory here
    ############################
    # not working on mac
    ############################
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts and args.instance_prompt is not None and not accelerator.device.type == "mps":
        del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
        free_memory()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    if not train_dataset.custom_instance_prompts and args.instance_prompt is not None:
        if not args.train_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds
            text_ids = instance_text_ids
            if args.with_prior_preservation:
                prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
                pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, class_pooled_prompt_embeds], dim=0)
                text_ids = torch.cat([text_ids, class_text_ids], dim=0)
        # if we're optimizing the text encoder (both if instance prompt is used for all images or custom prompts)
        # we need to tokenize and encode the batch prompts on all training steps
        else:
            tokens_one = tokenize_prompt(tokenizer_one, args.instance_prompt, max_sequence_length=77)
            tokens_two = tokenize_prompt(
                tokenizer_two, args.instance_prompt, max_sequence_length=args.max_sequence_length
            )
            if args.with_prior_preservation:
                class_tokens_one = tokenize_prompt(tokenizer_one, args.class_prompt, max_sequence_length=77)
                class_tokens_two = tokenize_prompt(
                    tokenizer_two, args.class_prompt, max_sequence_length=args.max_sequence_length
                )
                tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
                tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    if args.cache_latents:
        latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)

        if args.validation_prompt is None:
            del vae
            free_memory()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-flux-dev-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Prepare validation inputs
    validation_prompts_list = []
    validation_images_list = []
    validation_masks_list = [] 
    if args.validation_prompt:
        # Split by semicolon
        validation_prompts_list = [p.strip() for p in args.validation_prompt.split(";") if p.strip()]
    if args.validation_image:
        validation_images_list = [p.strip() for p in args.validation_image.split(";") if p.strip()]
    if args.validation_mask:
        validation_masks_list = [p.strip() for p in args.validation_mask.split(";") if p.strip()]

    # --- Input Broadcasting Logic ---
    inputs_present = [validation_prompts_list, validation_images_list]
    inputs_present.append(validation_masks_list)

    lists_with_multiple_items = [lst for lst in inputs_present if len(lst) > 1]

    if not lists_with_multiple_items:
        # All lists have 0 or 1 item
        max_len = max(len(lst) for lst in inputs_present) if inputs_present else 0
    else:
        first_multiple_len = len(lists_with_multiple_items[0])
        if any(len(lst) != first_multiple_len for lst in lists_with_multiple_items):
            raise ValueError(
                "Validation inputs (prompts, images, masks) have differing lengths > 1. "
                "Lengths must be 1 or match other inputs with length > 1. "
                f"Got lengths: prompts={len(validation_prompts_list)}, "
                f"images={len(validation_images_list)}, "
                f"masks={len(validation_masks_list) if not args.use_split_mask else 'N/A (split mask)'}"
            )
        max_len = first_multiple_len

    # Broadcast lists with single item
    if max_len > 0:
        if len(validation_prompts_list) == 1 and max_len > 1:
            validation_prompts_list = validation_prompts_list * max_len
        elif len(validation_prompts_list) == 0 and max_len > 0:
             raise ValueError("Validation prompts are required when validation images/masks are provided.")

        if len(validation_images_list) == 1 and max_len > 1:
            validation_images_list = validation_images_list * max_len
        elif len(validation_images_list) == 0 and max_len > 0:
             raise ValueError("Validation images are required when validation prompts are provided.")

        if len(validation_masks_list) == 1 and max_len > 1:
            validation_masks_list = validation_masks_list * max_len
        elif len(validation_masks_list) == 0 and max_len > 0 and not args.use_split_mask and args.mode == "mask":
                raise ValueError("Validation masks are required when validation prompts/images are provided and use_split_mask is False.")
    # --- End Broadcasting Logic ---

    num_validation_pairs = max_len
    if args.mode == "edit":
        for param in transformer.x_embedder.parameters():
            param.requires_grad = True
            print("set x_embedder.weight to trainable")
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            # set top parameter requires_grad = True for gradient checkpointing works
            accelerator.unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one])
            with accelerator.accumulate(models_to_accumulate):
                prompts = batch["prompts"]

                # encode batch prompts when custom prompts are provided for each image -
                if train_dataset.custom_instance_prompts or args.instance_prompt is None:
                    if not args.train_text_encoder:
                        prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                            prompts, text_encoders, tokenizers
                        )
                    else:
                        tokens_one = tokenize_prompt(tokenizer_one, prompts, max_sequence_length=77)
                        tokens_two = tokenize_prompt(
                            tokenizer_two, prompts, max_sequence_length=args.max_sequence_length
                        )
                        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=[None, None],
                            text_input_ids_list=[tokens_one, tokens_two],
                            max_sequence_length=args.max_sequence_length,
                            device=accelerator.device,
                            prompt=prompts,
                        )
                else:
                    elems_to_repeat = len(prompts)
                    if args.train_text_encoder:
                        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=[None, None],
                            text_input_ids_list=[
                                tokens_one.repeat(elems_to_repeat, 1),
                                tokens_two.repeat(elems_to_repeat, 1),
                            ],
                            max_sequence_length=args.max_sequence_length,
                            device=accelerator.device,
                            prompt=args.instance_prompt,
                        )

                # Convert images to latent space
                if args.cache_latents:
                    model_input = latents_cache[step].sample()
                else:
                    if args.mode == "edit":
                        pixel_values = batch["targets"].to(dtype=vae.dtype)
                    else:
                        pixel_values = batch["pixel_values"].to(dtype=vae.dtype)

                    model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                masked_image_latents = vae.encode(
                    batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype)
                ).latent_dist.sample()
                masked_image_latents = (masked_image_latents - vae.config.shift_factor) * vae.config.scaling_factor
                mask = batch["masks"]

                 # 5.resize mask to latents shape we we concatenate the mask to the latents
                mask = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
                mask = mask.view(
                    model_input.shape[0], model_input.shape[2], vae_scale_factor, model_input.shape[3], vae_scale_factor
                )  # batch_size, height, 8, width, 8
                mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
                mask = mask.reshape(
                    model_input.shape[0], vae_scale_factor * vae_scale_factor, model_input.shape[2], model_input.shape[3]
                )  
                latent_image_ids = FluxFillEditPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxFillEditPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                # handle guidance
                if accelerator.unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None
                    
               
                masked_image_latents = FluxFillEditPipeline._pack_latents(
                    masked_image_latents,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                mask = FluxFillEditPipeline._pack_latents(
                    mask,
                    batch_size=model_input.shape[0],
                    num_channels_latents=vae_scale_factor*vae_scale_factor,
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)
    
                transformer_input = torch.cat((packed_noisy_model_input, masked_image_latents), dim=2)    
                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=transformer_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = FluxFillEditPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute prior loss
                    prior_loss = torch.mean(
                        (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                            target_prior.shape[0], -1
                        ),
                        1,
                    )
                    prior_loss = prior_loss.mean()

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                if args.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(transformer.parameters(), text_encoder_one.parameters())
                        if args.train_text_encoder
                        else transformer.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # Increment global step FIRST
                global_step += 1
                # Update progress bar IMMEDIATELY after step increment
                progress_bar.update(1)
                trainable_param_names = []
                for name, param in transformer.named_parameters():
                    if param.requires_grad:
                        trainable_param_names.append(name)

                    # Checkpointing logic
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if args.train_embedder:
                        try:
                            unwrapped_transformer = accelerator.unwrap_model(transformer)
                            x_embedder_state_dict = unwrapped_transformer.x_embedder.state_dict()
                            x_embedder_save_path = Path(save_path) / "flux_x_embedder.safetensors"
                            safetensors.torch.save_file(x_embedder_state_dict, x_embedder_save_path)
                            logger.info(f"Saved x_embedder state to {x_embedder_save_path}")
                        except Exception as e:
                            logger.error(f"Failed to save x_embedder state during checkpointing: {e}")
                    logger.info(f"Saved state to {save_path}")

                # Validation logic
                if args.validation_prompt is not None and global_step % args.validation_steps == 0 and num_validation_pairs > 0:
                    logger.info(f"--- Starting validation at step {global_step} ---") # Log start

                    
                    if accelerator.device.type != "mps":
                        text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
                        tokenizer_one = CLIPTokenizer.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="tokenizer",
                            revision=args.revision,
                        )
                        tokenizer_two = T5TokenizerFast.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="tokenizer_2",
                            revision=args.revision,
                        )
                    
                    text_encoder_one.to(dtype=weight_dtype)
                    text_encoder_two.to(dtype=weight_dtype)
                    text_encoder_one.eval()
                    text_encoder_two.eval()

                    # Store images from all validation pairs
                    all_validation_images = []

                    # Dictionary to aggregate all validation logs for wandb
                    all_wandb_logs = {}
                    # List to aggregate images for tensorboard
                    all_tb_images = []
                    all_tb_keys = []
                    # Create pipeline
                    text_encoder_one.requires_grad_(False)
                    text_encoder_two.requires_grad_(False)
                    transformer.eval()
                    pipeline = FluxFillEditPipeline(
                        scheduler=noise_scheduler, vae=vae,
                        text_encoder=unwrap_model(text_encoder_one), tokenizer=tokenizer_one,
                        text_encoder_2 = unwrap_model(text_encoder_two), tokenizer_2 = tokenizer_two,
                        transformer=unwrap_model(transformer),

                    )
                    pipeline.to(accelerator.device,torch_dtype=weight_dtype)

                    logger.info(f"Running validation on {num_validation_pairs} pairs...")

                    for i in range(num_validation_pairs):
                        current_prompt = validation_prompts_list[i]
                        img_path = validation_images_list[i]

                        try:
                            val_image = load_image(img_path)
                        except Exception as e:
                            logger.error(f"Failed to load validation image {i+1}: {img_path}. Error: {e}")
                            continue # Skip this pair

                        if args.use_split_mask and not validation_masks_list and not args.mode == "edit":
                            # Run twice: left and right mask
                            mask_types = ["left", "right"] if not args.mask_only_right else ["right"]
                            for mask_type in mask_types:
                                wandb_data, images = log_validation(
                                    pipeline=pipeline, args=args, accelerator=accelerator,
                                    validation_prompt=current_prompt,
                                    val_image=val_image,
                                    val_mask=None, # Not needed for split mask
                                    mask_type=mask_type,
                                    validation_index=i,
                                    is_final_validation=False
                                )
                                all_validation_images.extend(images)    
                                all_wandb_logs.update(wandb_data)
                                all_tb_images.append(np.stack([np.asarray(img) for img in images]))
                                all_tb_keys.append(list(wandb_data.keys())[0])
                        elif args.mode == "edit":
                            # Run twice: left and right mask
                            mask_type = "white"
                            wandb_data, images = log_validation(
                                pipeline=pipeline, args=args, accelerator=accelerator,
                                validation_prompt=current_prompt,
                                val_image=val_image,
                                val_mask=None, # Not needed for split mask
                                mask_type=mask_type,
                                validation_index=i,
                                is_final_validation=False
                                )
                            all_validation_images.extend(images)    
                            all_wandb_logs.update(wandb_data)
                            all_tb_images.append(np.stack([np.asarray(img) for img in images]))
                            all_tb_keys.append(list(wandb_data.keys())[0])
                        else:
                            # Run once with provided mask
                            mask_path = validation_masks_list[i]
                            try:
                                val_mask = load_image(mask_path)
                            except Exception as e:
                                logger.error(f"Failed to load validation mask {i+1}: {mask_path}. Error: {e}")
                                continue # Skip this pair

                            wandb_data, images = log_validation(
                                pipeline=pipeline, args=args, accelerator=accelerator,
                                validation_prompt=current_prompt,
                                val_image=val_image,
                                val_mask=val_mask,
                                mask_type=None, # Use provided mask
                                validation_index=i,
                                is_final_validation=False
                            )
                            all_validation_images.extend(images)
                            all_wandb_logs.update(wandb_data)
                            all_tb_images.append(np.stack([np.asarray(img) for img in images]))
                            all_tb_keys.append(list(wandb_data.keys())[0])

                                        # --- Log aggregated results AFTER the loop ---
                    if all_wandb_logs:
                        print(f"DEBUG: Logging aggregated validation data for step: {global_step}")
                        accelerator.log(all_wandb_logs, step=global_step)

                    # Log to TensorBoard separately if needed
                    if all_tb_images:
                        for tracker in accelerator.trackers:
                            if tracker.name == "tensorboard":
                                for key, img_stack in zip(all_tb_keys, all_tb_images):
                                    # Log each pair's images under its unique key
                                    tracker.writer.add_images(key, img_stack, global_step, dataformats="NHWC")
                    # Cleanup pipeline
                    del pipeline
                    if accelerator.device.type != "mps":
                        del text_encoder_one, text_encoder_two
                    free_memory()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Set models back to train mode
                    transformer.train()
                    if args.train_text_encoder:
                        text_encoder_one.train()
                        accelerator.unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)

                    logger.info(f"--- Finished validation at step {global_step} ---") # Log end

            # Log loss and LR *after* potential validation
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step) # Log metrics with the correct global_step

            if global_step >= args.max_train_steps:
                break

        

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one.to(torch.float32))
        else:
            text_encoder_lora_layers = None

        FluxFillEditPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
        )

        # Final inference
        # Load previous pipeline
        pipeline = FluxFillEditPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        # load attention processors
        pipeline.load_lora_weights(args.output_dir)
        pipeline.to(accelerator.device) # Ensure pipeline is on the correct device

        # run inference
        final_images = []
        final_prompts_for_card = [] # Store unique prompts/info for card

        if args.validation_prompt and num_validation_pairs > 0:
            logger.info(f"Running final validation on {num_validation_pairs} pairs...")
            for i in range(num_validation_pairs):
                current_prompt = validation_prompts_list[i]
                img_path = validation_images_list[i]

                try:
                    val_image = load_image(img_path)
                except Exception as e:
                    logger.error(f"Failed to load final validation image {i+1}: {img_path}. Error: {e}")
                    continue

                if args.use_split_mask:
                    # Run twice: left and right mask    
                    mask_types = ["left", "right"] if not args.mask_only_right else ["right"]
                    for mask_type in mask_types:
                        wandb_data, images = log_validation(
                            pipeline=pipeline, args=args, accelerator=accelerator,
                            validation_prompt=current_prompt,
                            val_image=val_image, val_mask=None, mask_type=mask_type,
                            validation_index=i, is_final_validation=True
                        )
                        final_images.extend(images)
                        all_wandb_logs.update(wandb_data)
                        all_tb_images.append(np.stack([np.asarray(img) for img in images]))
                        all_tb_keys.append(list(wandb_data.keys())[0])
                        if images: # Add info for card only if images were generated
                             final_prompts_for_card.append(f"Pair {i} ({mask_type} mask): {current_prompt}")
                elif args.mode == "edit":
                    # Run once with provided mask
                    mask_type = "white"
                    wandb_data, images = log_validation(
                        pipeline=pipeline, args=args, accelerator=accelerator,
                        validation_prompt=current_prompt,
                        val_image=val_image, val_mask=None, mask_type=mask_type,
                        validation_index=i, is_final_validation=True
                    )
                else:
                    # Run once with provided mask
                    mask_path = validation_masks_list[i]
                    try:
                        val_mask = load_image(mask_path)
                    except Exception as e:
                        logger.error(f"Failed to load final validation mask {i+1}: {mask_path}. Error: {e}")
                        continue

                    wandb_data, images = log_validation(
                        pipeline=pipeline, args=args, accelerator=accelerator,
                        validation_prompt=current_prompt,
                        val_image=val_image, val_mask=val_mask, mask_type=None,
                        validation_index=i, is_final_validation=True
                    )
                    final_images.extend(images)
                    all_wandb_logs.update(wandb_data)
                    all_tb_images.append(np.stack([np.asarray(img) for img in images]))
                    all_tb_keys.append(list(wandb_data.keys())[0])
                    if images:
                        final_prompts_for_card.append(f"Pair {i}: {current_prompt}")

        # --- Log aggregated results AFTER the loop ---
        if all_wandb_logs:
            print(f"DEBUG: Logging aggregated validation data for step: {global_step}")
            accelerator.log(all_wandb_logs, step=global_step)

        # Log to TensorBoard separately if needed
        if all_tb_images:   
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    for key, img_stack in zip(all_tb_keys, all_tb_images):
                        # Log each pair's images under its unique key
                        tracker.writer.add_images(key, img_stack, global_step, dataformats="NHWC")

        if args.push_to_hub:
            # Use the first prompt from the list for the main card text for simplicity
            main_card_prompt = validation_prompts_list[0] if validation_prompts_list else args.instance_prompt
            save_model_card(
                repo_id,
                images=final_images, # Pass all generated images
                base_model=args.pretrained_model_name_or_path,
                train_text_encoder=args.train_text_encoder,
                instance_prompt=args.instance_prompt,
                validation_prompt=main_card_prompt, # Use first validation prompt for card text
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        images = None
        del pipeline

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
