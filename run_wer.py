import sys, os
from tqdm import tqdm
import multiprocessing
from jiwer import compute_measures
from zhon.hanzi import punctuation
import string
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import soundfile as sf
import scipy
import zhconv
from funasr import AutoModel
from datasets import load_dataset


punctuation_all = punctuation + string.punctuation + '.,!?'

wav_res_text_path = sys.argv[1]
res_path = sys.argv[2]
lang = sys.argv[3] # zh or en
device = "cuda:0"

def load_en_model():
    model_id = "/exp/leying.zhang/WenetSpeech4TTS/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model

def load_zh_model():
    model = AutoModel(model="paraformer-zh")
    return model

def process_one(hypo, truth):
    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        if x == '\'':
            continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')
    truth = truth.replace('[laughter]','')
    hypo = hypo.replace('[laughter]','')
    
    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')
    truth = truth.lower()
    hypo = hypo.lower()

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError

    measures = compute_measures(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures["wer"]
    subs = measures["substitutions"] / len(ref_list)
    dele = measures["deletions"] / len(ref_list)
    inse = measures["insertions"] / len(ref_list)
    
    # print("hyp", hypo,  "truth", truth, "WER", wer, "subs", subs, "dele", dele, "inse", inse)
    return (truth, hypo, wer, subs, dele, inse)


def run_asr(wav_res_text_path, res_path):
    if lang == "en":
        processor, model = load_en_model()
    elif lang == "zh":
        model = load_zh_model()

    params = []
    with open (wav_res_text_path, "r") as f:
        lines = f.readlines()
    for line in tqdm(lines):
        line = line.strip()
        assert len(line.split('\t'))==4, f"Error: Line does not have exactly 4 tab-separated parts: {line}"
        synthesized_speech, gt_speech, prompt_speech, gt_text = line.split('\t')
        params.append((synthesized_speech, gt_text))
    
    fout = open(res_path, "w")
    n_higher_than_50 = 0
    wers_below_50 = []
    for synthesized_speech, gt_text in tqdm(params):
        if lang == "en":
            wav, sr = sf.read(synthesized_speech)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            
            # result = pipe(wav)
            input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device)
            # forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
            # model.generation_config=forced_decoder_ids
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        elif lang == "zh":
            res = model.generate(input=synthesized_speech,
                    batch_size_s=300)
            transcription = res[0]["text"]
            transcription = zhconv.convert(transcription, 'zh-cn')

        raw_truth, raw_hypo, wer, subs, dele, inse = process_one(transcription, gt_text)
        fout.write(f"{synthesized_speech}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n")
        fout.flush()

run_asr(wav_res_text_path, res_path)
