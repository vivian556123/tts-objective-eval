import sys, os
from tqdm import tqdm

metalst = sys.argv[1]
wav_dir = sys.argv[2]
wav_res_ref_text = sys.argv[3]

f = open(metalst)
lines = f.readlines()
f.close()

with open(wav_res_ref_text, 'w') as f_w:
    for line in tqdm(lines):
        # print("line", line, line.strip().split('\t'))
        if len(line.strip().split('\t')) == 3:
            utt, prompt_wav, infer_text = line.strip().split('\t')
            utt = os.path.basename(utt)
        else: 
            print("Error in processing line", line)
        
        if not os.path.exists(os.path.join(wav_dir, utt)):
            print("os.path.join(wav_dir, utt )", os.path.join(wav_dir, utt ))
            # continue

        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)

        # if not os.path.isabs(infer_wav):
        #     infer_wav = os.path.join(os.path.dirname(metalst), infer_wav)

        out_line = '\t'.join([os.path.join(wav_dir, utt ), prompt_wav, infer_text])
        f_w.write(out_line + '\n')

print("successfully write pairs into ", wav_res_ref_text)
