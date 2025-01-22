import sys, os
from tqdm import tqdm

metalst = sys.argv[1]
synthesized_dir = sys.argv[2] 
ground_truth_dir = sys.argv[3]
prompt_dir = sys.argv[4]
final_metafile_for_evaluation = sys.argv[5]
generated_wav_suffix = sys.argv[6]
    
f = open(metalst)
lines = f.readlines()
f.close()

with open(final_metafile_for_evaluation, 'w') as f_w:
    for line in tqdm(lines):
        if len(line.strip().split('\t')) == 3:
            gt_speech, prompt_speech, gt_text = line.strip().split('\t')
        elif len(line.strip().split('\t')) == 2:
            gt_speech, prompt_speech = line.strip().split('\t')
            gt_text = "None"
        elif len(line.strip().split('\t')) == 1:
            gt_speech = line.strip()
            prompt_speech = "None"
            gt_text = "None"
        else: 
            print("Error in processing line", line)
        utt_basename = os.path.basename(gt_speech)
            
        if os.path.exists(ground_truth_dir):
            gt_speech = os.path.join(ground_truth_dir, utt_basename)
        
        if os.path.exists(synthesized_dir): # the suffix only works for the generated speech
            synthesized_speech = os.path.join(synthesized_dir, utt_basename.split('.')[0]+generated_wav_suffix)
        
        if os.path.exists(prompt_dir):
            prompt_speech = os.path.join(prompt_dir, utt_basename)
<<<<<<< HEAD
            
        if not os.path.exists(gt_speech) or not os.path.exists(synthesized_speech) or not os.path.exists(prompt_speech):
            print("the speech does not exist!", "gt_speech", gt_speech, "synthesized_speech", synthesized_speech, "prompt_speech", prompt_speech)
            continue
        
=======
            if not os.path.exists(prompt_speech):
                print("the prompt speech does not exist!", "prompt_speech", prompt_speech)
            
        if not os.path.exists(gt_speech) or not os.path.exists(synthesized_speech):
            print("the speech does not exist!", "gt_speech", gt_speech, "synthesized_speech", synthesized_speech)
       
>>>>>>> d80f3fbe88ebcbaef2199278842e1ca4162f5737

        out_line = '\t'.join([synthesized_speech, gt_speech, prompt_speech, gt_text])
        f_w.write(out_line + '\n')

print("successfully write pairs into ", final_metafile_for_evaluation)
