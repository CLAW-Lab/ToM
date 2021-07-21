import torch
from googletrans import Translator
from varname import nameof

import pickle
import tqdm

import argparse

from multiprocessing import Pool
from functools import reduce

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str)
parser.add_argument("--split", type=str)
parser.add_argument("--threads", type=int)

args = parser.parse_args()

translator = Translator()

train_cap = torch.load("coco/labs/train_org")
valid_cap = torch.load("coco/labs/valid_org")
test_cap = torch.load("coco/labs/test_org")

i2w = torch.load("coco/dics/i2w")
w2i = torch.load("coco/dics/w2i")

cnt = 0

languages = ["de", "lt", "zh-cn", "it", "fr", "pt", "es", "ja", "el"]
splits = ["train_cap", "valid_cap", "test_cap"]

assert args.lang in languages and args.split in splits
language = args.lang
split = args.split
print(f"Now start translating {language}")
collection = locals()[split]

def get_translated(end_points):
    begin, end = end_points
    caps = []
    for image in collection[begin:end]:
        image_res_caps = []
        for cap in image:
            cap = cap[1:-1]
            try:
                cap.remove(w2i["<UNK>"])
            except:
                pass
            try:
                cap.remove(w2i["&apos;s"])
            except:
                pass
            cap = ' '.join(map(lambda x: i2w[x], cap))
            try:
                translated_cap = translator.translate(cap, dest=language).text
            except:
                translated_cap = ""
            image_res_caps.append(translated_cap)
        caps.append(image_res_caps)
    return caps

partition = []
for i in range(args.threads):
    if i != args.threads-1:
        partition.append((len(collection)//args.threads * i, len(collection)//args.threads * (i+1)))
    else:
        partition.append((len(collection)//args.threads * i, len(collection)))

import time
start = time.time()
with Pool(args.threads) as p:
    caps_to_merge = p.map(get_translated, partition)

caps = reduce(lambda x, y: x + y, caps_to_merge)
end = time.time()
print(f"Multiprocessing uses {end - start} seconds.")

pickle.dump(caps, open(f"coco/labs/{split}_{language}.pkl", "wb"))
        


