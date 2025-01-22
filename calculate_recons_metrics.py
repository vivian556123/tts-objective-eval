import tqdm
import argparse
import os
import torch
import fire
import torch.nn.functional as F
from torchaudio.transforms import Resample
import librosa
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as trans
import sys
from pathlib import Path
import itertools
import logging
import shutil
import subprocess
import warnings
from collections import OrderedDict
from distutils.util import strtobool
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile
from pystoi.stoi import stoi
from pesq import pesq
import pandas as pd
from scipy.io import wavfile

import scipy.stats
from scipy.signal import butter, sosfilt

import torch

from pesq import pesq
from pystoi import stoi
import librosa

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', type=str)
    parser.add_argument('--scores', type=str)
    parser.add_argument('--target_sr', type=int, default=16000)
    parser.add_argument('--pesq_mode', type=str, default='wb')
    args = parser.parse_args()

    f = open(args.pair)
    lines = f.readlines()
    f.close()

    synthesized_list = []
    ground_truth_list = []
    for line in lines:
        assert len(line.strip().split('\t'))==4, f"Error: Line does not have exactly 4 tab-separated parts: {line}"
        synthesized_speech, gt_speech, prompt_speech, gt_text = line.strip().split('\t')
        synthesized_list.append(synthesized_speech)
        ground_truth_list.append(gt_speech)
        
    scores_w = open(args.scores, 'w')
    assert len(synthesized_list) == len(ground_truth_list)

    pesq_list = []
    estoi_list = []
    stoi_list = []
    for t1, t2 in tqdm.tqdm(zip(synthesized_list, ground_truth_list), total=len(synthesized_list)):
        t1_path = t1.strip()
        t2_path = t2.strip()
        
        if not os.path.exists(t1_path) or not os.path.exists(t2_path):
            print("t1_path", t1_path, "t2_path", t2_path, " in mode ", args.mode, " not exist")
            continue
        

        test_signal, sr = librosa.load(t1_path, sr = args.target_sr)      # 待评估音频
        ref_signal, sr = librosa.load(t2_path, sr = args.target_sr)  # 参考音频

        pesq = pesq(sr, ref_signal, test_signal, args.pesq_mode)
        estoi = stoi(ref_signal, test_signal, sr, extended=True)
        stoi = stoi(ref_signal, test_signal, sr, extended=False)
        
        scores_w.write(f'{t1_path}\t{t2_path}\t{pesq}\t{estoi}\t{stoi}\n')
        pesq_list.append(pesq)
        estoi_list.append(estoi)
        stoi_list.append(stoi)
        scores_w.flush()
        
    scores_w.write(f'avg PESQ score between generated speech and grount truth in {args.pesq_mode} is : {sum(pesq_list)/len(pesq_list)}\n')
    scores_w.write(f'avg estoi between generated speech and grount truth: {sum(estoi_list)/len(estoi_list)}\n')
    scores_w.write(f'avg stoi between generated speech and grount truth: {sum(stoi_list)/len(stoi_list)}\n')
    print(f'avg pesq  between generated speech and grount truth: {sum(pesq_list)/len(pesq_list)}')
    print(f'avg estoi between generated speech and grount truth: {sum(estoi_list)/len(estoi_list)}')
    print(f'avg stoi between generated speech and grount truth: {sum(stoi_list)/len(stoi_list)}')
    scores_w.flush()
    
with open(os.path.join(os.path.dirname(args.scores),"total_results"),'a') as total_result:
    total_result.write(f"mode: {args.mode} PESQ: {sum(pesq_list)/len(pesq_list)} ESTOI: {sum(estoi_list)/len(estoi_list)} STOI: {sum(stoi_list)/len(stoi_list)}  \n")