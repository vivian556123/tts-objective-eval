#!/bin/bash

# Set default values if arguments are not provided
meta_lst="default_meta_lst"
synthesized_dir="default_synthesized_dir"
lang="en"
generated_wav_suffix=".wav"
prompt_dir="default_prompt_dir"
ground_truth_dir="default_ground_truth_dir"

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
    --lang)
      lang="$2"
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
      ground_truth_dir="$2"
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
out_score_file="$synthesized_dir/mcd_results_score"

# Get the working directory
workdir=$(cd "$(dirname "$0")"; cd ../; pwd)

# Define the Python command (modify as needed for your environment)
# python_command="srun -p a10,4090 --gres=gpu:1 --mem 40G --qos qlong -c 2 python"
python_command="python"

# Execute the Python scripts with the provided or default arguments
$python_command get_wav_res_ref_text.py $meta_lst $synthesized_dir $ground_truth_dir $prompt_dir $wav_res_ref $generated_wav_suffix
$python_command calculate_mcd.py --pair $wav_res_ref  --scores $out_score_file 

