# InFluxEdit
# Flux Fill LoRA Editing

This project explores image editing with **FLUX.1 Fill [dev]** by leveraging (a) its powerful outpainting and inpainting capabilities, and (b) using its additional channels in a slightly modified way. Our method enables high-fidelity image editing while avoiding full model finetuning, building on prior efforts such as **ACE++**.

## 🧠 Key Idea

We investigate two methods for using **Flux Fill [dev]** to generate training data for image editing LoRAs:

### a) **Spatial Concatenation**

Concatenate the **input image** and **generated output image** side-by-side and use a mask to supervise the right (output) half. This approach was previously used by the AI community with LoRAs.

![Spatial Concatenation](https://github.com/andjoer/InFluxEdit/raw/main/graphics/concat_spacial.png)

### b) **Extra Channel Adaption**

Feed only the **input image** to Flux Fill and use a white mask, but do not apply the mask on the input image itself (i.e., do not zero out masked pixel values as is standard with Flux Fill). This retains reference context and boosts output fidelity. Unlike ACE++, we do not fully finetune the model when changing the purpose of the channels.

![Extra Channel Masking](https://github.com/andjoer/InFluxEdit/raw/main/graphics/concat_channels.png)

---

## 💡 Method Summary

- The model uses **Flux Fill [dev]** inpainting mode with an all-white mask in the extra channel.
- LoRA adapters are trained with this setup—**no full model finetune is needed**.
- Optionally, the **`x_embedder`** module (which encodes extra input channels) can be set as trainable.

---

## 🔬 Findings (Preliminary)

1. **LoRA training alone is sufficient** for high-quality editing using extra channel input.
2. **Training `x_embedder` is optional** – it may help in some tasks but doesn’t harm performance.
3. **Spatial concatenation leads to lower fidelity** results compared to processing the unmasked input.

---

## 📦 Dataset Structure

The dataset organization depends on the training mode:

### Spatial Concatenation Mode
- Input images are pre-concatenated with target images side-by-side into single images
- Training data:  contains the concatenated images
- Validation data: follows same concatenated format
- Optional prompts: directory with text files containing per-image prompts.
- Optional masks: directory with mask images (mandatory if not using `--use_split_mask`)

### Edit Mode  
- Input and target images are kept separate
- Training data: 
  - input images
  - contains corresponding target images
- Validation data:
  - input images
- Optional prompts: directory with text files containing per-image prompts.

### Important Notes
- For inference, both modes expect only input images (no pre-concatenation needed)
- The inference scripts handle spatial concatenation internally when needed
- Prompts are required if `--instance_prompt` is not defined
- Masks are only used in concatenation mode when `--use_split_mask` is disabled

## 📦 Example Dataset

This project includes scripts to easily download, prepare, and train on the synthetic "Fill Circles" dataset.

**Steps:**

1.  **Prepare the Dataset:**
    First, run the preparation script. This will download the dataset (if not already present), combine the input/target images for spatial concatenation mode, and extract the prompts into individual `.txt` files.
    ```bash
    bash prep_fill_circles_ds.sh
    ```
    This script creates the `data/my_fill50k` directory containing `train_A` (input outlines), `train_B` (filled circles), `train_combined` (side-by-side images), and `prompts` for the spatial and edit modes.
    Make sure to export your wandb and huggingface tokens before running the script.

2.  **Choose a Training Mode and Run:**

    *   **a) Spatial Concatenation Mode:**
        This mode uses the side-by-side combined images (`train_combined`) and learns to fill the right half based on the left half and the prompt. Use the `train_circles_combined.sh` script.
        ```bash
        bash train_circles_combined.sh
        ```

    *   **b) Edit Mode:**
        This mode uses the original outline image (`train_A`) as input and the filled circle image (`train_B`) as the target, applying a white mask internally. Use the `train_circles_edit.sh` script.
        ```bash
        bash train_circles_edit.sh
        ```

---

## 🎜️ License

This work uses FLUX.1 Fill [dev], which is distributed under the [FLUX.1 [dev] Non-Commercial License](https://bfl.ml).

---

## 📊 Acknowledgments

- **FLUX.1 Fill [dev]**: for cutting-edge image completion and inpainting.
- **The Huggingface Team**: for the original dreambooth lora training script.
- **ACE++**: for exploring the adaption of the extra channels.
- **Sebastian-Zok**: to our knowledge, the first to release a training script for Flux Fill LoRAs. This project borrows from his work.

# InFluxEdit Training Script Documentation

**train_influxedit.py** fine-tunes LoRA adapters for image editing with FLUX.1 Fill [dev] without full model finetuning. The script supports two primary editing modes:

- **Mask Mode ("fill")**  
  In this mode, a provided or synthetic mask determines which image regions are processed. When using the `--use_split_mask` flag, synthetic masks are generated by splitting the image into left/right halves. (Setting `--mask_only_right` forces only right-half masks.)

- **Edit Mode ("edit")**  
  A white mask is applied to the input image while a separate target image is used to preserve reference context. In edit mode, the extra channel encoder (`x_embedder`) can optionally be trained using `--train_embedder`.

The training data may be loaded from local directories or a Hugging Face dataset. Custom column mappings allow you to specify which dataset fields correspond to images, masks, targets, and prompts. Optionally, per-image prompts can be provided via a designated prompt directory.

---

## Command-Line Arguments

The following tables list all parsed arguments (except those related to prior preservation) organized by topic. Each entry shows the argument name, a brief description, and its default value.

### 1. Model & Dataset Configuration

| Argument                            | Description                                                                  | Default Value |
|-------------------------------------|------------------------------------------------------------------------------|---------------|
| `--pretrained_model_name_or_path`   | Path or model identifier for the pretrained base model. (Required)           | None          |
| `--revision`                        | Model revision to use (if applicable).                                       | None          |
| `--variant`                         | Variant of model files (e.g., "fp16").                                       | None          |
| `--dataset_name`                    | Name or path of a Hugging Face dataset for instance images.                  | None          |
| `--dataset_config_name`             | Configuration name for the dataset (if applicable).                        | None          |

### 2. Data Directories & Dataset Columns

| Argument                | Description                                                                                      | Default Value |
|-------------------------|--------------------------------------------------------------------------------------------------|---------------|
| `--instance_data_dir`   | Directory containing instance (training) images.                                                | None          |
| `--mask_data_dir`       | Directory containing mask images.                                                               | None          |
| `--target_data_dir`     | Directory containing target images (used in edit mode).                                         | None          |
| `--caption_data_dir`    | Directory with text prompt files (each file named after its corresponding image with a `.txt` extension). | None          |
| `--cache_dir`           | Directory where downloaded models/datasets will be cached.                                      | None          |
| `--image_column`        | Dataset column name for images (when using a dataset).                                           | image         |
| `--caption_column`      | Dataset column name for per-image prompts.                                                       | None          |
| `--mask_column`         | Dataset column name for mask images.                                                             | None          |
| `--target_column`       | Dataset column name for target images (used in edit mode when using a dataset).                  | None          |

### 3. Training Modes & Data Augmentation

| Argument                | Description                                                                                      | Default Value |
|-------------------------|--------------------------------------------------------------------------------------------------|---------------|
| `--mode`                | Training mode: `"mask"` (fill/inpainting) or `"edit"`.                                           | mask          |
| `--train_embedder`      | Train the extra channel embedder (`x_embedder`) (applicable in edit mode).                         | False         |
| `--train_text_encoder`  | Whether to train the text encoder.                                                               | False         |
| `--repeats`             | Number of times to repeat the training data.                                                     | 1             |
| `--instance_prompt`     | Default prompt for instance images if no custom prompt is provided.                              | None          |
| `--resolution`          | Target resolution(s) for input images (single value or comma-separated list for random cropping).  | 512           |
| `--center_crop`         | Use center crop (instead of random crop) after resizing images.                                  | False         |
| `--random_flip`         | Randomly flip images horizontally as an augmentation.                                           | False         |
| `--size`                | Target size for images; a single integer or a comma-separated list (e.g., "256,512,768").           | 512           |

### 4. Validation Options

| Argument                  | Description                                                                                       | Default Value |
|---------------------------|---------------------------------------------------------------------------------------------------|---------------|
| `--validation_prompt`     | Prompt(s) used during validation (comma-separated if multiple).                                   | None          |
| `--validation_image`      | File path(s) to validation image(s).                                                              | None          |
| `--validation_mask`       | File path(s) to validation mask(s); ignored if `--use_split_mask` is set.                          | None          |
| `--num_validation_images` | Number of images to generate during each validation run.                                          | 1             |
| `--validation_steps`      | Frequency (in training steps) to run validation.                                                  | 50            |
| `--validation_width`      | Width of generated validation images.                                                             | 1024          |
| `--validation_height`     | Height of generated validation images.                                                            | 1024          |

### 5. Mask Options

| Argument              | Description                                                                                              | Default Value |
|-----------------------|----------------------------------------------------------------------------------------------------------|---------------|
| `--use_split_mask`    | Enable synthetic split mask generation (creates left/right masks automatically).                         | False         |
| `--mask_only_right`   | When using split masks, generate only right-half masks.                                                | False         |

### 6. Text Sequence Settings

| Argument                 | Description                                                       | Default Value |
|--------------------------|-------------------------------------------------------------------|---------------|
| `--max_sequence_length`  | Maximum sequence length for the T5 text encoder.                  | 512           |

### 7. Training Hyperparameters

| Argument                          | Description                                                                  | Default Value |
|-----------------------------------|------------------------------------------------------------------------------|---------------|
| `--train_batch_size`              | Training batch size per device.                                              | 4             |
| `--sample_batch_size`             | Batch size for sampling images.                                              | 4             |
| `--num_train_epochs`              | Number of training epochs.                                                   | 1             |
| `--max_train_steps`               | Total number of training steps (overrides epoch count if set).               | None          |
| `--gradient_accumulation_steps`   | Number of steps over which gradients are accumulated before an update.       | 1             |
| `--gradient_checkpointing`        | Enable gradient checkpointing to reduce memory usage.                        | False         |
| `--seed`                          | Random seed for reproducible training.                                       | None          |

### 8. Optimizer & Learning Rate Settings

| Argument                         | Description                                                                                          | Default Value |
|----------------------------------|------------------------------------------------------------------------------------------------------|---------------|
| `--learning_rate`                | Initial learning rate (after any warmup).                                                            | 1e-4          |
| `--guidance_scale`               | Guidance scale factor for conditioning (e.g., in guidance-distilled models).                         | 3.5           |
| `--text_encoder_lr`              | Learning rate for the text encoder (if trained).                                                     | 5e-6          |
| `--scale_lr`                     | Scale the learning rate by the number of GPUs, accumulation steps, and batch size.                   | False         |
| `--lr_scheduler`                 | Learning rate scheduler type (options: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup). | constant      |
| `--lr_warmup_steps`              | Number of warmup steps for the learning rate scheduler.                                              | 500           |
| `--lr_num_cycles`                | Number of cycles for cosine_with_restarts scheduler.                                                 | 1             |
| `--lr_power`                     | Power factor for the polynomial scheduler.                                                          | 1.0           |
| `--optimizer`                    | Optimizer type to use (choose between "AdamW" and "prodigy").                                          | AdamW         |
| `--use_8bit_adam`                | Use 8-bit Adam (from bitsandbytes); applicable only if optimizer is AdamW.                           | False         |
| `--adam_beta1`                   | Beta1 parameter for the optimizer.                                                                   | 0.9           |
| `--adam_beta2`                   | Beta2 parameter for the optimizer.                                                                   | 0.999         |
| `--prodigy_beta3`                | Coefficient for computing the Prodigy step size (if not provided, uses sqrt(beta2)).                  | None          |
| `--prodigy_decouple`             | Use AdamW-style decoupled weight decay in the Prodigy optimizer.                                       | True          |
| `--adam_weight_decay`            | Weight decay for main model parameters (e.g., UNet parameters).                                       | 1e-4          |
| `--adam_weight_decay_text_encoder` | Weight decay for the text encoder parameters.                                                     | 1e-3          |
| `--lora_layers`                  | Comma-separated list of transformer modules to apply LoRA (e.g., "attn.to_k,attn.to_q,attn.to_v").      | None          |
| `--adam_epsilon`                 | Epsilon value for the optimizer.                                                                     | 1e-8          |
| `--prodigy_use_bias_correction`  | Enable bias correction in the Prodigy optimizer.                                                     | True          |
| `--prodigy_safeguard_warmup`     | Remove LR from the denominator during warmup in Prodigy to safeguard the step size.                  | True          |
| `--max_grad_norm`                | Maximum norm for gradient clipping.                                                                  | 1.0           |

### 9. Output, Logging & Checkpointing

| Argument                        | Description                                                                                        | Default Value    |
|---------------------------------|----------------------------------------------------------------------------------------------------|------------------|
| `--output_dir`                   | Directory where checkpoints and outputs will be saved.                                             | flux-dreambooth-lora |
| `--checkpointing_steps`          | Frequency (in steps) to save a training checkpoint.                                                | 500              |
| `--checkpoints_total_limit`      | Maximum number of checkpoints to store.                                                            | None             |
| `--resume_from_checkpoint`       | Resume training from a checkpoint (path or `"latest"`).                                            | None             |
| `--push_to_hub`                  | Push the final model to the Hugging Face Hub.                                                      | False            |
| `--hub_token`                    | Token used for uploading the model to the Hub.                                                     | None             |
| `--hub_model_id`                 | Repository name on the Hub to sync with the local output directory.                                | None             |
| `--logging_dir`                  | Directory for TensorBoard logs.                                                                    | logs             |
| `--report_to`                    | Logging integration (e.g., "tensorboard", "wandb", "comet_ml", or "all").                          | tensorboard      |

### 10. Mixed Precision & Advanced Settings

| Argument                        | Description                                                                                       | Default Value |
|---------------------------------|---------------------------------------------------------------------------------------------------|---------------|
| `--allow_tf32`                  | Allow TF32 on Nvidia Ampere GPUs for faster training.                                             | False         |
| `--cache_latents`               | Cache VAE latents to speed up training.                                                           | False         |
| `--mixed_precision`             | Mixed precision mode: "no", "fp16", or "bf16".                                                    | None          |
| `--upcast_before_saving`        | Upcast trainable transformer layers to float32 before saving.                                     | False         |
| `--dataloader_num_workers`      | Number of subprocesses for data loading.                                                          | 0             |
| `--weighting_scheme`            | Weighting scheme for loss computation (options: sigma_sqrt, logit_normal, mode, cosmap, none).      | none          |
| `--logit_mean`                  | Mean value for the logit_normal weighting scheme.                                                 | 0.0           |
| `--logit_std`                   | Standard deviation for the logit_normal weighting scheme.                                         | 1.0           |
| `--mode_scale`                  | Scale factor for the mode weighting scheme.                                                       | 1.29          |
| `--local_rank`                  | Local rank for distributed training.                                                              | -1            |

---

## Training Workflow

1. **Initialization & Accelerator Setup**  
   The script initializes the Hugging Face Accelerator to support distributed and mixed-precision training. Random seeds are set for reproducibility, and logging is configured (via wandb, TensorBoard, etc.).

2. **Data Loading & Preprocessing**  
   Data is loaded either from local directories or a Hugging Face dataset. If using local data, instance images, masks, and (optionally) custom prompt files (from `--caption_data_dir`) are read. When `--use_split_mask` is enabled, synthetic masks are automatically generated (left/right or right-only based on `--mask_only_right`).

3. **Model Components & LoRA Adaptation**  
   A VAE encodes images into latent space, and a transformer (FluxTransformer2DModel) processes these latents concatenated with mask (or target) information. Two text encoders condition the model on text prompts. LoRA adapters are applied to designated transformer modules (as specified by `--lora_layers`), and optionally the text encoder is trained if enabled.

4. **Forward Pass & Loss Computation**  
   Input images are encoded, noise is added via a scheduler, and the resulting noisy latent representations (with mask or target concatenated) are processed by the transformer. The loss is computed as the mean squared error between the predicted and actual noise.

5. **Optimization & Checkpointing**  
   Gradients are accumulated according to `--gradient_accumulation_steps` and clipped using `--max_grad_norm`. Checkpoints are saved at intervals defined by `--checkpointing_steps` and training may resume from a saved checkpoint via `--resume_from_checkpoint`.

6. **Validation**  
   At intervals defined by `--validation_steps`, validation is performed using the specified `--validation_prompt` and other validation settings. In split mask mode, synthetic masks are generated automatically for validation.

7. **Saving & Hub Upload**  
   At the end of training, the LoRA weights (and optionally the extra channel encoder if trained) are saved. If enabled, the final model is pushed to the Hugging Face Hub.

---

## Example Commands

### Example 1: Spatial Concatenation (Mask Mode with Split Masks)

This example trains the model using spatial concatenation. Instance images are loaded from `$INSTANCE_DIR`, synthetic split masks are enabled, and per-image prompts are loaded from `$PROMPT_DIR`.

```bash
accelerate launch train_influxedit.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --use_split_mask \
  --rank=16 \
  --output_dir=$OUTPUT_DIR \
  --caption_data_dir=$PROMPT_DIR \
  --resolution=704,1024 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --checkpointing_steps=100 \
  --max_train_steps=100000 \
  --validation_prompt="validation prompt" \
  --validation_image="" \
  --validation_steps=50 \
  --validation_width=1024 \
  --validation_height=512 \
  --num_validation_images=1 \
  --mixed_precision="bf16" \
  --gradient_checkpointing
```

### Example 2: Edit Mode Training

In edit mode, the script uses a separate target image directory (`$TARGET_DIR`) and applies a white mask to the instance image to preserve reference context. Custom prompts are loaded from `$PROMPT_DIR`.

```bash
accelerate launch train_influxedit.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --target_data_dir=$TARGET_DIR \
  --caption_data_dir=$PROMPT_DIR \
  --mode="edit" \
  --rank=64 \
  --output_dir=$OUTPUT_DIR \
  --resolution=1024,1536,2048 \
  --train_batch_size=2 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=2 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --checkpointing_steps=100 \
  --max_train_steps=100000 \
  --validation_prompt="validation prompt" \
  --validation_image="" \
  --validation_steps=100 \
  --validation_width=1024 \
  --validation_height=1024 \
  --num_validation_images=1 \
  --mixed_precision="bf16" \
  --gradient_checkpointing
```

*Replace environment variables (e.g., `$MODEL_NAME`, `$INSTANCE_DIR`, `$TARGET_DIR`, `$PROMPT_DIR`, `$OUTPUT_DIR`) with your actual values.*

---

## Conclusion

**train_influxedit.py** provides a flexible and configurable training pipeline for fine-tuning image editing models with LoRA adapters using FLUX.1 Fill [dev]. By choosing between mask (fill) and edit modes, configuring synthetic split mask generation, and specifying data directories, dataset columns, and training hyperparameters, users can tailor the training process to a wide range of editing tasks—all without full model finetuning.

Happy training!


