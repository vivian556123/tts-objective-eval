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
import numpy as np

# Get the absolute path of the current file's directory
current_dir = Path(__file__).resolve().parent
# Get the parent directory
parent_dir = current_dir.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))
from speaker_verification.models.ecapa_tdnn import ECAPA_TDNN_SMALL
from speaker_verification.models.utils import UpstreamExpert


MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]



''' Res2Conv1d + BatchNorm1d + ReLU
'''


class Res2Conv1dReluBn(nn.Module):
    '''
    in_channels == out_channels == channels
    '''

    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)

        return out


''' Conv1d + BatchNorm1d + ReLU
'''


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


''' The SE connection of 1D case.
'''


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)

        return out


''' SE-Res2Block of the ECAPA-TDNN architecture.
'''


# def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
#     return nn.Sequential(
#         Conv1dReluBn(channels, 512, kernel_size=1, stride=1, padding=0),
#         Res2Conv1dReluBn(512, kernel_size, stride, padding, dilation, scale=scale),
#         Conv1dReluBn(512, channels, kernel_size=1, stride=1, padding=0),
#         SE_Connect(channels)
#     )


class SE_Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, scale, se_bottleneck_dim):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(out_channels, kernel_size, stride, padding, dilation, scale=scale)
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)

        return x + residual


''' Attentive weighted mean and standard deviation pooling.
'''


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, attention_channels, kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x):

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x_in))
        # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


def init_model(model_name, checkpoint=None):
    
    if model_name == 'unispeech_sat':
        config_path = 'config/unispeech_sat.th'
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='unispeech_sat', config_path=config_path)
    elif model_name == 'wavlm_base_plus':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=config_path)
    elif model_name == 'wavlm_large':
        config_path = os.path.dirname(checkpoint)
        print("Begin initializing WavLM Large")
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
    elif model_name == 'hubert_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=config_path)
    elif model_name == 'wav2vec2_xlsr':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=config_path)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')


    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model


def verification_similarity(model,  wav1, wav2, use_gpu=True, checkpoint=None, 
                            wav1_start_sr=0, wav2_start_sr=0, wav1_end_sr=-1, wav2_end_sr=-1, 
                            device="cuda:0"):

    wav1_path = wav1
    wav1, sr1 = librosa.load(wav1, sr=None, mono=False)

    # wav1, sr1 = sf.read(wav1)
    if len(wav1.shape) == 2:
        wav1 = wav1[:,0]
    # wav2, sr2 = sf.read(wav2)
    wav2, sr2 = librosa.load(wav2, sr=None, mono=False)
    if len(wav2.shape) == 2:
        wav2 = wav2[0,:]    # wav2.shape: [channels, T]

    wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
    wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    resample2 = Resample(orig_freq=sr2, new_freq=16000)    
    wav1 = resample1(wav1)
    wav2 = resample2(wav2)
    
    wav1 = wav1[...,wav1_start_sr:wav1_end_sr if wav1_end_sr > 0 else wav1.shape[-1]]
    wav2 = wav2[...,wav2_start_sr:wav2_end_sr if wav2_end_sr > 0 else wav2.shape[-1]]

    if str(device) != 'cpu':
        wav1 = wav1.to(device)
        wav2 = wav2.to(device)

    with torch.no_grad():
        emb1 = model(wav1)
        emb2 = model(wav2)

    sim = F.cosine_similarity(emb1, emb2)
    # print("The similarity score between two audios is {:.4f} (-1.0, 1.0).".format(sim[0].item()))
    return sim


def extract_embedding(model_name,  wav1, use_gpu=True, checkpoint=None, wav1_start_sr=0, wav1_end_sr=-1, model=None, device="cuda:0"):

    assert model_name in MODEL_LIST, 'The model_name should be in {}'.format(MODEL_LIST)
    model = init_model(model_name, checkpoint) if model is None else model

    wav1, sr1 = sf.read(wav1)
    wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    wav1 = resample1(wav1)
    wav1 = wav1[...,wav1_start_sr:wav1_end_sr if wav1_end_sr > 0 else wav1.shape[-1]]
    if str(device) != 'cpu':
        model = model.to(device)
        wav1 = wav1.to(device)

    model.eval()
    with torch.no_grad():
        emb1 = model(wav1)
    return emb1, model



if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('pair')
    parser.add_argument('--model_name')
    parser.add_argument('--checkpoint')
    parser.add_argument('--scores')
    parser.add_argument('--wav1_start_sr', type=int)
    parser.add_argument('--wav2_start_sr', type=int)
    parser.add_argument('--wav1_end_sr', type=int)
    parser.add_argument('--wav2_end_sr', type=int)
    parser.add_argument('--device', default="cuda")

    
    
    args = parser.parse_args()

    f = open(args.pair)
    lines = f.readlines()
    f.close()

    tsv1 = []
    tsv2 = []
    
    for line in lines:
        assert len(line.strip().split('\t'))==4, f"Error: Line does not have exactly 4 tab-separated parts: {line}"
        synthesized_speech, gt_speech, prompt_speech, gt_text = line.strip().split('\t')
        
        tsv1.append(synthesized_speech)
        tsv2.append(prompt_speech)

    scores_w = open(args.scores, 'w')
    assert len(tsv1) == len(tsv2)

    score_list = []
    
    assert args.model_name in MODEL_LIST, 'The model_name should be in {}'.format(MODEL_LIST)
    model = init_model(args.model_name, args.checkpoint)
    model.to(args.device)
    model.eval()

    for t1, t2 in tqdm.tqdm(zip(tsv1, tsv2), total=len(tsv1)):
        t1_path = t1.strip()
        t2_path = t2.strip()
        if not os.path.exists(t1_path) or not os.path.exists(t2_path):
            print("t1_path", t1_path, "t2_path", t2_path, "not exists")
            continue

        sim = verification_similarity(model, t1_path, t2_path, use_gpu=True, checkpoint=args.checkpoint, 
                                      wav1_start_sr=args.wav1_start_sr, wav2_start_sr=args.wav2_start_sr, 
                                      wav1_end_sr=args.wav1_end_sr, wav2_end_sr=args.wav2_end_sr, 
                                      device=args.device)

        if sim is None:
            continue
        scores_w.write(f'{t1_path}\t{t2_path}\t{sim.cpu().item()}\n')
        score_list.append(sim.cpu().item())
        scores_w.flush()
    scores_w.write(f'avg sim score between generated speech and it\'s corresponding prompt: {sum(score_list)/len(score_list)}')
    exp_path = os.path.dirname(args.scores)
    print(f'{exp_path } avg sim score between generated speech and it\'s corresponding prompt: {sum(score_list)/len(score_list)}')
    scores_w.flush()
    
    with open(os.path.join(os.path.dirname(args.scores),"total_results"),'a') as total_result:
        total_result.write(f"SIM: {sum(score_list)/len(score_list)} \n")

