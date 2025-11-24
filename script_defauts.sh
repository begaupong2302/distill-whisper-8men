#!/usr/bin/env bash

TRAIN_NAME="../data4whipser+../data4whipser+../data4whipser+../data4whipser+../data4whipser+../data4whipser+../data4whipser+../data4whipser+../data4whipser+../data4whipser+../data4whipser+../data4whipser+../data4whipser+../data4whipser"
TRAIN_CONFIG_NAME="de+en+vi+fr+ko+ja+zh+de+en+vi+fr+ko+ja+zh"
TRAIN_SPLIT_NAME="train+train+train+train+train+train+train+validation+validation+validation+validation+validation+validation+validation"
EVAL_NAME="../data4whipser+../data4whipser+../data4whipser+../data4whipser+../data4whipser+../data4whipser+../data4whipser"
EVAL_CONFIG_NAME="de+en+vi+fr+ko+ja+zh"
EVAL_SPLIT_NAME="test+test+test+test+test+test+test"

accelerate launch run_distillation.py \
  --model_name_or_path "../distil-base-v3-init" \
  --teacher_model_name_or_path "openai/whisper-large-v3" \
  --train_dataset_name "$TRAIN_NAME" \
  --train_dataset_config_name "$TRAIN_CONFIG_NAME" \
  --train_split_name "$TRAIN_SPLIT_NAME" \
  --audio_column_name "audio" \
  --text_column_name "text+text+text+text+text+text+text+text+text+text+text+text+text+text" \
  --train_dataset_samples "1+1+1+1+1+1+1+1+1+1+1+1+1+1" \
  --eval_dataset_name "$EVAL_NAME" \
  --eval_dataset_config_name "$EVAL_CONFIG_NAME" \
  --eval_split_name "$EVAL_SPLIT_NAME" \
  --audio_column_name "audio" \
  --eval_text_column_name "text+text+text+text+text+text+text" \
  --eval_steps 1000 \
  --save_steps 1000 \
  --warmup_steps 50 \
  --learning_rate 0.0001 \
  --lr_scheduler_type "constant_with_warmup" \
  --timestamp_probability 0.2 \
  --condition_on_prev_probability 0.2 \
  --task "transcribe" \
  --logging_steps 25 \
  --save_total_limit 1 \
  --max_steps 5000 \
  --wer_threshold 20 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --dataloader_num_workers 1 \
  --preprocessing_num_workers 8 \
  --ddp_timeout 7200 \
  --dtype "bfloat16" \
  --attn_implementation "flash_attention_2" \
  --output_dir "../results" \
  --do_train \
  --do_eval \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --predict_with_generate \
  --freeze_encoder \
  --freeze_embed_positions \
  --streaming True \
  --use_pseudo_labels False
  # --push_to_hub

# thêm kl divergence tăng wer do kiến thức distill được áp dụng cho cả word-level lẫn sequence-lvel
