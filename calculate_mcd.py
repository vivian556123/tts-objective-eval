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
from pymcd.mcd import Calculate_MCD



def cal_mcd(mcd_plain_toolbox, mcd_dtw_toolbox, mcd_dtw_sl_toolbox,  wav1, wav2):
    plain_mcd_value = mcd_plain_toolbox.calculate_mcd(wav1, wav2)
    dtw_mcd_value = mcd_dtw_toolbox.calculate_mcd(wav1, wav2)
    dtw_sl_mcd_value = mcd_dtw_sl_toolbox.calculate_mcd(wav1, wav2)
    return plain_mcd_value, dtw_mcd_value, dtw_sl_mcd_value



if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('pair')
    parser.add_argument('--scores')
    args = parser.parse_args()

    f = open(args.pair)
    lines = f.readlines()
    f.close()

    tsv1 = []
    tsv2 = []
    for line in lines:
        e = line.strip().split('\t')
        if len(e) == 4:
            part1, _, _, part2 = line.strip().split('\t')
        elif len(e) == 3:
            part1, part2, _ = line.strip().split('\t')
        else:
            part1, part2 = line.strip().split('\t')[:2]
        # print("part1", part1, "part2", part2)
        tsv1.append(part1)
        tsv2.append(part2)

    scores_w = open(args.scores, 'w')
    assert len(tsv1) == len(tsv2)

    mcd_plain_toolbox = Calculate_MCD(MCD_mode="plain")
    mcd_dtw_toolbox = Calculate_MCD(MCD_mode="dtw")
    mcd_dtw_sl_toolbox = Calculate_MCD(MCD_mode="dtw")
    
    plain_mcd_list = []
    dtw_mcd_list = []
    dtw_sl_mcd_list = []
    for t1, t2 in tqdm.tqdm(zip(tsv1, tsv2), total=len(tsv1)):
        t1_path = t1.strip()
        t2_path = t2.strip()
        if not os.path.exists(t1_path) or not os.path.exists(t2_path):
            print("t1_path", t1_path, "t2_path", t2_path, "not exists")
            continue
        try:
            plain_mcd_value, dtw_mcd_value, dtw_sl_mcd_value = cal_mcd(mcd_plain_toolbox, mcd_dtw_toolbox, mcd_dtw_sl_toolbox,  t1_path, t2_path)
        except Exception as e:
            print(str(e))
            continue

        scores_w.write(f'{t1_path}\t{t2_path}\t{plain_mcd_value}\t{dtw_mcd_value}\t{dtw_sl_mcd_value}\n')
        plain_mcd_list.append(plain_mcd_value)
        dtw_mcd_list.append(dtw_mcd_value)
        dtw_sl_mcd_list.append(dtw_sl_mcd_value)
        scores_w.flush()
    scores_w.write(f'avg plain mcd score between generated speech and grount truth: {sum(plain_mcd_list)/len(plain_mcd_list)}')
    scores_w.write(f'avg dtw mcd score between generated speech and grount truth: {sum(dtw_mcd_list)/len(dtw_mcd_list)}')
    scores_w.write(f'avg dtw sl mcd score between generated speech and grount truth: {sum(dtw_sl_mcd_list)/len(dtw_sl_mcd_list)}')
    print(f'avg mcd score between generated speech and grount truth: {sum(plain_mcd_list)/len(plain_mcd_list)}')
    print(f'avg mcd score between generated speech and grount truth: {sum(dtw_mcd_list)/len(dtw_mcd_list)}')
    print(f'avg mcd score between generated speech and grount truth: {sum(dtw_sl_mcd_list)/len(dtw_sl_mcd_list)}')
    scores_w.flush()
    
with open(os.path.join(os.path.dirname(args.scores),"total_results"),'w') as total_result:
    total_result.write(f"MCD PLAIN: {sum(plain_mcd_list)/len(plain_mcd_list)} DTW: {sum(dtw_mcd_list)/len(dtw_mcd_list)} DTW_SL: {sum(dtw_sl_mcd_list)/len(dtw_sl_mcd_list)}  \n")