import torch
from collections import defaultdict
from typing import List


lang_set = torch.load("lang_id.data")


def language_identification(sent: List[int]) -> int:
    result = set(range(10))
    for word in sent:
        if len(lang_set[word]):
            result = result.intersection(lang_set[word])
    try:
        return list(result)[0]
    except:
        return None


if __name__ == "__main__":
    data = torch.load("coco_ml/labs/train_org_lang")
    lang_set = defaultdict(set)
    for image in data:
        for lang_id, lang in enumerate(image):
            for sent in lang:
                for word in sent:
                    lang_set[word].add(lang_id)
    torch.save(lang_set, "lang_id.data")