import os
from tqdm import tqdm
root = '/home/ubuntu11/zzc/data/senti/PERR/data'
for id in range(7):
    guess=[]
    for vid in tqdm(os.listdir(root)):
        if vid.split('_')[id] not in guess:
            guess.append(vid.split('_')[id])
    print(id,len(guess))