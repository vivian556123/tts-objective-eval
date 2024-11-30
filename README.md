## Requirements
To install all dependencies, run 
```
python -m pip install pip==24.0
pip3 install -r requirements.txt
```

## Metrics
The word error rate (WER) and speaker similarity (SIM) metrics are adopted for objective evaluation. 
* For WER, we employ [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) and [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh) as the automatic speech recognition (ASR) engines for English and Mandarin, respectively.
* For SIM, we use WavLM-large fine-tuned on the speaker verification task ([model link](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view)) to obtain speaker embeddings used to calculate the cosine similarity of speech samples of each test utterance against reference clips.
* For MCD, we use py_mcd
* For PESQ, we use pesq
* For STOI and ESTOI, we use pystoi

## Data configuration Preparation
Meta-data format: 
```
ground_truth_path1'\t'prompt_path1'\t'ground_truth_text1
ground_truth_path2'\t'prompt_path2'\t'ground_truth_text2
ground_truth_path3'\t'prompt_path3'\t'ground_truth_text3
```

## Utilization

#### WER
```
bash cal_wer.sh {--meta_lst the path of the meta file} {--synthesized_dir the directory of synthesized audio} {--prompt_dir the directory of prompt audio} {--ground_truth_dir the directory of ground truth audio} {--lang: zh or en} {--generated_wav_suffix the suffix of the generated speech}
```

#### SIM
```
bash cal_sim.sh {--meta_lst the path of the meta file} {--synthesized_dir the directory of synthesized audio} {--prompt_dir the directory of prompt audio} {--ground_truth_dir the directory of ground truth audio} {--generated_wav_suffix the suffix of the generated speech}  {--checkpoint path/wavlm_large_finetune.pth}
```

#### MCD
```
bash cal_mcd.sh {--meta_lst the path of the meta file} {--synthesized_dir the directory of synthesized audio} {--prompt_dir the directory of prompt audio} {--ground_truth_dir the directory of ground truth audio} {--generated_wav_suffix the suffix of the generated speech} 
```

#### PESQ,ESTOI,STOI
```
bash cal_recons_metrics.sh {--meta_lst the path of the meta file} {--synthesized_dir the directory of synthesized audio} {--prompt_dir the directory of prompt audio} {--ground_truth_dir the directory of ground truth audio} {--generated_wav_suffix the suffix of the generated speech} 
```


## Acknowledgement

This repo is inspired by and borrows many codes from  [Seed-TTS-Eval](https://github.com/BytedanceSpeech/seed-tts-eval) and [UniSpeech](https://github.com/microsoft/UniSpeech). Thank you to the authors of these projects for their contributions!

