#!/bin/bash

# Set default values if arguments are not provided
meta_lst="default_meta_lst"
synthesized_dir="default_synthesized_dir"
generated_wav_suffix=".wav"
prompt_dir="default_prompt_dir"
ground_truth_dir="default_ground_truth_dir"
checkpoint_path="/exp/leying.zhang/pretrained_models/wavlm_large_finetune.pth"

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --meta_lst)
      meta_lst="$2"
      shift 2
      ;;
    --synthesized_dir)
      synthesized_dir="$2"
      shift 2
      ;;
    --generated_wav_suffix)
      generated_wav_suffix="$2"
      shift 2
      ;;
    --ground_truth_dir)
      ground_truth_dir="$2"
      shift 2
      ;;
    --prompt_dir)
      prompt_dir="$2"
      shift 2
      ;;
    --checkpoint_path)
      checkpoint_path="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Define file paths based on the arguments
wav_res_ref="$synthesized_dir/wav_res_ref_text"
out_score_file="$synthesized_dir/utmos_results_score"

# Get the working directory
python_command=python
$python_command get_wav_res_ref_text.py $meta_lst $synthesized_dir $ground_truth_dir $prompt_dir $wav_res_ref $generated_wav_suffix

workdir=$(cd "$(dirname "$0")"; pwd)
cd $workdir/thirdparty/UTMOS-demo
$python_command predict.py \
    --mode predict_meta \
    --inp_path  $wav_res_ref \
    --out_path  $out_score_file \
