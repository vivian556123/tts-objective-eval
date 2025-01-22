#!/bin/bash

# Set default values if arguments are not provided
meta_lst="default_meta_lst"
synthesized_dir="default_synthesized_dir"
<<<<<<< HEAD
lang="en"
generated_wav_suffix=".wav"
prompt_dir="default_prompt_dir"
ground_truth_dir="default_ground_truth_dir"
checkpoint_path="/exp/leying.zhang/pretrained_models/wavlm_large_finetune.pth"
=======
generated_wav_suffix=".wav"
prompt_dir="default_prompt_dir"
ground_truth_dir="default_ground_truth_dir"
checkpoint_path="/data/v-leyizhang/pretrained_model/wavlm_large_finetune.pth"
>>>>>>> d80f3fbe88ebcbaef2199278842e1ca4162f5737

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
<<<<<<< HEAD
    --lang)
      lang="$2"
      shift 2
      ;;
=======
>>>>>>> d80f3fbe88ebcbaef2199278842e1ca4162f5737
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
out_score_file="$synthesized_dir/sim_results_score"

# Get the working directory
python_command=python
$python_command get_wav_res_ref_text.py $meta_lst $synthesized_dir $ground_truth_dir $prompt_dir $wav_res_ref $generated_wav_suffix

workdir=$(cd "$(dirname "$0")"; pwd)
cd $workdir/thirdparty/UniSpeech/downstreams/speaker_verification/
$python_command verification_pair_list_v2.py $wav_res_ref \
    --model_name wavlm_large \
    --checkpoint $checkpoint_path \
    --scores $out_score_file \
    --wav1_start_sr 0 \
    --wav2_start_sr 0 \
    --wav1_end_sr -1 \
    --wav2_end_sr -1 \
    --device cuda 