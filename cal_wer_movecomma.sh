set -x

conda activate eval

meta_lst=$1
output_dir=$2
lang=$3

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.wer

workdir=$(cd $(dirname $0); cd ../; pwd)

# python_command="srun -p a10,4090 --gres=gpu:1 --mem 40G --qos qlong -c 2 python"
# python_command="srun -p a10,4090 --gres=gpu:1 --mem 40G --qos qlong -c 2 python"
python_command="python"

$python_command get_wav_res_ref_text.py $meta_lst $output_dir $wav_wav_text

out_score_file=$output_dir/wer_results

$python_command run_wer.py $wav_wav_text $out_score_file\_score $lang

$python_command average_wer.py $out_score_file\_score $out_score_file
