# seed-tts-eval
:boom: This repository contains the objective test set as proposed in our project, [seed-TTS](https://arxiv.org/abs/2406.02430), along with the scripts for metric calculations.  Due to considerations for AI safety, we will NOT be releasing the source code and model weights of seed-TTS. We invite you to experience the speech generation feature within ByteDance products. :boom:

To evaluate the zero-shot speech generation ability of our model, we propose an out-of-domain objective evaluation test set. This test set consists of samples extracted from English (EN) and Mandarin (ZH) public corpora that are used to measure the model's performance on various objective metrics. Specifically, we employ 1,000 samples from the [Common Voice](https://commonvoice.mozilla.org/en) dataset and 2,000 samples from the [DiDiSpeech-2](https://arxiv.org/pdf/2010.09275) dataset. 

## Requirements
To install all dependencies, run 
```
pip3 install -r requirements.txt
```

## Metrics
The word error rate (WER) and speaker similarity (SIM) metrics are adopted for objective evaluation. 
* For WER, we employ [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) and [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh) as the automatic speech recognition (ASR) engines for English and Mandarin, respectively.
* For SIM, we use WavLM-large fine-tuned on the speaker verification task ([model link](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view)) to obtain speaker embeddings used to calculate the cosine similarity of speech samples of each test utterance against reference clips.
* For MCD, we use py_mcd

## Data configuration Preparation
Meta-data format: ground_truth_path'\t'prompt_path'\t'ground_truth_text

for example /home/leying.zhang/code/noise-robust-tts/LibriTTS-metadata/libritts_test.csv

## Utilization
```
# WER
bash cal_wer.sh {the path of the meta file} {the directory of synthesized audio} {language: zh or en} {the suffix of the generated speech}
# SIM
bash cal_sim.sh {the path of the meta file} {the directory of synthesized audio} {path/wavlm_large_finetune.pth: /exp/leying.zhang/pretrained_models/wavlm_large_finetune.pth } {the suffix of the generated speech}
# MCD
bash cal_mcd.sh {the path of the meta file} {the directory of synthesized audio} {mode: clean} {the suffix of the generated speech}
# PESQ, SISNR, SDR
bash cal_enh_metrics.sh {the path of the meta file} {the directory of synthesized audio} {mode: clean} {the suffix of the generated speech}


```
