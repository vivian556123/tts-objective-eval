set -x

conda activate eval

meta_lst=$1
output_dir=$2
checkpoint_path=$3

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.wer

# python3 get_wav_res_ref_text.py $meta_lst $output_dir $output_dir/wav_res_ref_text

workdir=$(cd $(dirname $0); pwd)

cd $workdir/thirdparty/UniSpeech/downstreams/speaker_verification/

out_score_file=$output_dir/sim_results

# python_command="srun -p a10,4090 --gres=gpu:1 --mem 40G --qos qlong -c 2 python"
python_command="python"

$python_command verification_pair_list_v2.py $output_dir/wav_res_ref_text \
    --model_name wavlm_large \
    --checkpoint $checkpoint_path \
    --scores $out_score_file\_score \
    --wav1_start_sr 0 \
    --wav2_start_sr 0 \
    --wav1_end_sr -1 \
    --wav2_end_sr -1 \
    --device cuda
