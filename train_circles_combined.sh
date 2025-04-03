export MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
export INSTANCE_DIR="data/my_fill50k/train_combined"
export PROMPT_DIR="data/my_fill50k/prompts_combined"
export OUTPUT_DIR="flux-fill-circles-combined-lora"

python train_influxedit.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --caption_data_dir=$PROMPT_DIR \
  --output_dir=$OUTPUT_DIR \
  --use_split_mask \
  --mask_only_right \
  --rank=16 \
  --resolution=512 \
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
  --validation_prompt="Before and after image. On the left half a cricle and on the right side the very same circle like this: violet circle with orange background" \
  --validation_image="data/my_fill50k/test_combined/40000.png" \
  --validation_steps=50 \
  --validation_width=512 \
  --validation_height=256 \
  --num_validation_images=2 