#!/usr/bin/env bash

BASE="../data4whipser/"

LANGUAGE=("de" "en" "vi" "fr" "ko" "ja" "zh")

TRAIN_NAME=""

TRAIN_SPLIT_NAME="train+train+train+train+train+train+train+validation+validation+validation+validation+validation+validation+validation"

EVAL_NAME=""

EVAL_SPLIT_NAME="test+test+test+test+test+test+test"

for i in {1..2}; do
    for p in "${LANGUAGE[@]}"; do
        FULL_PATH="$BASE$p"
        if [ -z "$TRAIN_NAME" ]; then
            TRAIN_NAME="$FULL_PATH"
        else
            TRAIN_NAME="$TRAIN_NAME+$FULL_PATH"
        fi

        if [ "$i" -eq 1 ]; then
            if [ -z "$EVAL_NAME" ]; then
                EVAL_NAME="$FULL_PATH"
            else
                EVAL_NAME="$EVAL_NAME+$FULL_PATH"
            fi
        fi
    done
done

accelerate launch run_distillation.py \
  --model_name_or_path "../distil-base-v3-init" \
  --teacher_model_name_or_path "openai/whisper-medium" \
  --train_dataset_name "$TRAIN_NAME" \
  --train_split_name "$TRAIN_SPLIT_NAME" \
  --audio_column_name "audio_path" \
  --text_column_name "text+text+text+text+text+text+text+text+text+text+text+text+text+text" \
  --train_dataset_samples "1+1+1+1+1+1+1+1+1+1+1+1+1+1" \
  --eval_dataset_name "$EVAL_NAME" \
  --eval_split_name "$EVAL_SPLIT_NAME" \
  --audio_column_name "audio_path" \
  --eval_text_column_name "text+text+text+text+text+text+text" \
  --eval_steps 1000 \
  --save_steps 1000 \
  --warmup_steps 500 \
  --learning_rate 0.0001 \
  --lr_scheduler_type "linear" \
  --timestamp_probability 0.5 \
  --condition_on_prev_probability 0.2 \
  --task "transcribe" \
  --logging_steps 100 \
  --save_total_limit 1 \
  --max_steps 20000 \
  --wer_threshold 20 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 8 \
  --dataloader_num_workers 1 \
  --preprocessing_num_workers 8 \
  --ddp_timeout 7200 \
  --dtype "bfloat16" \
  --attn_implementation "flash_attention_2" \ #sdpa
  --output_dir "../results" \
  --do_train \
  --do_eval \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --predict_with_generate \
  --streaming True \
  --use_pseudo_labels False
  # --push_to_hub

# thêm kl divergence tăng wer do kiến thức distill được áp dụng cho cả word-level lẫn sequence-level
