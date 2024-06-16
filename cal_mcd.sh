set -x

meta_lst=$1
output_dir=$2

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.wer

# python3 get_wav_res_ref_text.py $meta_lst $output_dir $output_dir/wav_res_ref_text

out_score_file=$output_dir/mcd_results

python_command="srun -p a10,4090 --gres=gpu:1 --mem 40G --qos qlong -c 2 python"

$python_command calculate_mcd.py $output_dir/wav_res_ref_text  --scores $out_score_file\_score 

