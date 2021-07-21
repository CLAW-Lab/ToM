import os
import pickle
from collections import OrderedDict
import itertools
import random

import numpy as np
import torch
import torch.nn.functional as F

from typing import Tuple, List, Dict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def rejection_sampling(target_sample, seed_train_images, seed_train_captions, left_samples, batch_size):
#     T = 30 # rejection sampling temperature

#     def overlap_sents(lst1, lst2):
#         set1 = set(lst1)
#         set2 = set(lst2)
#         return len(set1 & set2) / len(set1 | set2)

#     def overlap(captions_0, captions_1):
#         overlap_sum = 0
#         for caption_0 in captions_0:
#             for caption_1 in captions_1:
#                 overlap_sum += overlap_sents(caption_0, caption_1)
#         return overlap_sum / len(captions_0) / len(captions_1)

#     def unnorm_prob(a_id, b_id):
#         # return torch.exp(-torch.dist(seed_train_images[a_id], seed_train_images[b_id])/T)
#         return torch.exp(overlap(seed_train_captions[a_id], seed_train_captions[b_id])/T)

#     distrs = []
#     distrs_app = distrs.append
#     for i in range(batch_size):
#         distr = random.choice(left_samples)
#         while(distr == target_sample[i] or unnorm_prob(target_sample[i], distr) < random.random()):
#             distr = random.choice(left_samples)
#         distrs_app(distr)
#     return distrs


# def get_s2p_batch(seed_train_images, seed_train_captions, args, batch_size=None, dist_matrix=None):
#     data_size = seed_train_images.shape[0]
#     if batch_size is None:
#         batch_size = args.s2p_batch_size
#     if batch_size > data_size:
#         batch_size = data_size

#     target_sample = random.sample(list(range(data_size)), batch_size)
#     left_samples = list(range(data_size))
#     distractor_samples = [[] for i in range(args.num_distrs)]
#     for i in range(batch_size):
#         if dist_matrix is None:
#             dsample = np.random.choice(left_samples, args.num_distrs, replace=False)
#         else:
#             dsample = np.random.choice(left_samples, args.num_distrs, replace=False,
#                                         p=dist_matrix[target_sample[i]])
#         for j in range(args.num_distrs):
#             distractor_samples[j].append(dsample[j])

#     image_batch = seed_train_images[target_sample]
#     distractor_image_batch = []
#     for ds in distractor_samples:
#         d_image_batch = seed_train_images[ds]
#         distractor_image_batch.append(d_image_batch)
#     caption_batch = [seed_train_captions[t] for t in target_sample]
#     caption_len_batch = torch.tensor([len(c) for c in caption_batch], dtype=torch.long, device=device)

#     caption_batch = torch.tensor([np.pad(c, (0, args.seq_len - len(c))) for c in caption_batch], dtype=torch.long, device=device)

#     caption_batch_onehot = torch.zeros(caption_batch.shape[0], caption_batch.shape[1], args.vocab_size,
#                              device=device).scatter_(-1, caption_batch.unsqueeze(-1), 1)

#     return image_batch, distractor_image_batch, caption_batch_onehot, caption_len_batch


def get_s2p_batch(seed_train_images, seed_train_captions, args, all_seed_train_captions=None, batch_size=None):
    data_size = seed_train_images.shape[0]
    if batch_size is None:
        batch_size = args.s2p_batch_size
    if batch_size > data_size:
        batch_size = data_size

    target_sample = random.sample(list(range(data_size)), batch_size)
    left_samples = list(range(data_size))
    distractor_samples = []
    for i in range(args.num_distrs):
        dsample = random.sample(left_samples, batch_size)
        distractor_samples.append(dsample)

    image_batch = seed_train_images[target_sample]
    distractor_image_batch = []
    for ds in distractor_samples:
        d_image_batch = seed_train_images[ds]
        distractor_image_batch.append(d_image_batch)
    caption_batch = [seed_train_captions[t] for t in target_sample]
    if all_seed_train_captions is not None:
        all_caption_batch = [all_seed_train_captions[t] for t in target_sample]
    caption_len_batch = torch.tensor(
        [len(c) for c in caption_batch], dtype=torch.long, device=device)

    caption_batch = torch.tensor([np.pad(c, (0, args.seq_len - len(c)))
                                  for c in caption_batch], dtype=torch.long, device=device)

    caption_batch_onehot = torch.zeros(caption_batch.shape[0], caption_batch.shape[1], args.vocab_size,
                                       device=device).scatter_(-1, caption_batch.unsqueeze(-1), 1)

    if all_seed_train_captions is not None:
        return image_batch, distractor_image_batch, caption_batch_onehot, caption_len_batch, all_caption_batch
    else:
        return image_batch, distractor_image_batch, caption_batch_onehot, caption_len_batch


def get_batch_with_speaker(train_images, speaker, args, batch_size=None):
    data_size = train_images.shape[0]
    if batch_size is None:
        batch_size = args.s2p_batch_size
    if batch_size > data_size:
        batch_size = data_size

    target_sample = random.sample(list(range(data_size)), batch_size)
    left_samples = list(range(data_size))
    distractor_samples = []
    for i in range(args.num_distrs):
        dsample = random.sample(left_samples, batch_size)
        distractor_samples.append(dsample)

    image_batch = train_images[target_sample]
    distractor_image_batch = []
    for ds in distractor_samples:
        d_image_batch = train_images[ds]
        distractor_image_batch.append(d_image_batch)

    train_captions, train_captions_len = speaker.forward(image_batch)
    train_captions = train_captions.detach()
    train_captions_len = train_captions_len.detach()

    return image_batch, distractor_image_batch, train_captions, train_captions_len


def get_distractor_images(image_pool, args, batch_size=None):
    data_size = image_pool.shape[0]
    if batch_size is None:
        batch_size = data_size
    left_samples = list(range(data_size))
    distractor_samples = []
    for i in range(args.num_distrs):
        dsample = random.sample(left_samples, batch_size)
        distractor_samples.append(dsample)

    distractor_image_batch = []
    for ds in distractor_samples:
        d_image_batch = image_pool[ds]
        distractor_image_batch.append(d_image_batch)

    return distractor_image_batch


# def get_pop_batch(train_images, args, batch_size=None, dist_matrix=None):
#     data_size = train_images.shape[0]
#     if batch_size is None:
#         batch_size = args.pop_batch_size

#     target_sample = random.sample(list(range(data_size)), batch_size)
#     left_samples = list(range(data_size))
#     distractor_samples = [[] for i in range(args.num_distrs)]
#     for i in range(batch_size):
#         if dist_matrix is None:
#             dsample = np.random.choice(left_samples, args.num_distrs, replace=False)
#         else:
#             dsample = np.random.choice(left_samples, args.num_distrs, replace=False,
#                                        p=dist_matrix[target_sample[i]])
#         for j in range(args.num_distrs):
#             distractor_samples[j].append(dsample[j])

#     image_batch = train_images[target_sample]
#     distractor_image_batch = []
#     for ds in distractor_samples:
#         d_image_batch = train_images[ds]
#         distractor_image_batch.append(d_image_batch)

#     return image_batch, distractor_image_batch

def get_pop_batch(train_images, args, batch_size=None):
    data_size = train_images.shape[0]
    if batch_size is None:
        batch_size = args.pop_batch_size

    target_sample = random.sample(list(range(data_size)), batch_size)
    left_samples = list(range(data_size))
    distractor_samples = []
    for i in range(args.num_distrs):
        dsample = random.sample(left_samples, batch_size)
        distractor_samples.append(dsample)

    image_batch = train_images[target_sample]
    distractor_image_batch = []
    for ds in distractor_samples:
        d_image_batch = train_images[ds]
        distractor_image_batch.append(d_image_batch)

    return image_batch, distractor_image_batch


def trim_caps(caps, minlen, maxlen):
    new_cap = [[cap for cap in cap_i if maxlen >=
                len(cap) >= minlen] for cap_i in caps]
    return new_cap


def truncate_dicts(w2i, i2w, trunc_size):
    symbols_to_keep = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    inds_to_keep = [w2i[s] for s in symbols_to_keep]

    w2i_trunc = OrderedDict(itertools.islice(w2i.items(), trunc_size))
    i2w_trunc = OrderedDict(itertools.islice(i2w.items(), trunc_size))

    for s, i in zip(symbols_to_keep, inds_to_keep):
        w2i_trunc[s] = i
        i2w_trunc[i] = s

    return w2i_trunc, i2w_trunc


def truncate_captions(train_captions, valid_captions, test_captions, w2i, i2w):
    unk_ind = w2i["<UNK>"]

    def truncate_data(data):
        for i in range(len(data)):
            for ii in range(len(data[i])):
                for iii in range(len(data[i][ii])):
                    if data[i][ii][iii] not in i2w:
                        data[i][ii][iii] = unk_ind
        return data

    train_captions = truncate_data(train_captions)
    valid_captions = truncate_data(valid_captions)
    test_captions = truncate_data(test_captions)

    return train_captions, valid_captions, test_captions


def load_model(model_dir, model, device):
    model_dicts = torch.load(os.path.join(
        model_dir, 'model.pt'), map_location=device)
    model.load_state_dict(model_dicts)
    iters = model_dicts['iters']
    best_test_acc = model_dicts['test_acc']
    print("Best Test acc:", best_test_acc, " at", iters, "iters")


def save_to_file(vals, folder='', file=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_str = os.path.join(folder, file+'.pkl')
    with open(save_str, 'wb') as f1:
        pickle.dump(vals, f1)


def torch_save_to_file(to_save, folder='', file=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(to_save, os.path.join(folder, file))


def to_sentence(inds_list, i2w, trim=False):
    sentences = []
    for inds in inds_list:
        if type(inds) is not list:
            inds = list(inds)
        sentence = []
        for i in inds:
            sentence.append(i2w[i])
            if i2w[i] == "<PAD>" and trim:
                break
        sentences.append(' '.join(sentence))
    return sentences


def filter_caps(captions, images, w2i, perc, keep_caps=False,
                original_captions=None):
    if keep_caps:
        original_captions = captions
    new_train_captions = []
    new_train_images = []
    all_new_train_captions = []
    for ci, cap in enumerate(captions):
        random.shuffle(cap)
        for cap_ in cap:
            if len(cap) > 0 and cap_.count(w2i["<UNK>"]) / len(cap_) < perc:
                new_train_captions.append(cap_)
                new_train_images.append(images[ci])
                if keep_caps:
                    all_new_train_captions.append(original_captions[ci])
                break
    if keep_caps:
        return new_train_captions, torch.stack(new_train_images), all_new_train_captions
    else:
        return new_train_captions, torch.stack(new_train_images)


def sample_gumbel(shape, eps=1e-20):
    U = torch.empty(shape, device=device).uniform_(0, 1)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temp):
    y = (logits + sample_gumbel(logits.shape)) / temp
    return F.softmax(y, dim=-1)


def gumbel_softmax(logits, temp, hard):
    y = gumbel_softmax_sample(logits, temp)
    y_max, y_max_idx = torch.max(y, 1, keepdim=True)
    if hard:
        y_hard = torch.zeros(y.shape, device=device).scatter_(1, y_max_idx, 1)
        y = (y_hard - y).detach() + y
    return y, y_max_idx


def sample(logits: torch.Tensor, temp: float = 1.0) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    dists = torch.distributions.Categorical(logits=logits * temp)
    result = dists.sample()
    logprob = dists.log_prob(result)
    return result, logprob


def truncate_msg(msg, msg_lens):
    result_msg = []
    for i, j in zip(msg, msg_lens):
        result_msg.append(i[1:j+1])
    return result_msg


def build_vocab(word_lang_list: dict, word_lang_dist: dict) -> set:
    vocab = set()
    for lang in word_lang_list:
        vocab.update(word_lang_list[lang][:word_lang_dist[lang]])
    return vocab


def update_vocab(w2i: dict, i2w: dict, vocab: set) -> Tuple[dict, dict]:
    symbols_to_keep = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    new_i2w = dict()
    new_w2i = dict()
    i2i = dict()
    for idx, w in enumerate(itertools.chain(symbols_to_keep, vocab)):
        assert i2w[w2i[w]] == w
        i2i[w2i[w]] = idx
        new_w2i[w] = idx
        new_i2w[idx] = w
    return new_w2i, new_i2w, i2i


def index_map(data, i2i, unk_ind):
    for i in range(len(data)):
        for ii in range(len(data[i])):
            for iii in range(len(data[i][ii])):
                if data[i][ii][iii] not in i2i:
                    data[i][ii][iii] = unk_ind
                else:
                    data[i][ii][iii] = i2i[data[i][ii][iii]]
    return data


def sample_vocab_dirichlet(alpha: float, whole_ratio: float,
                           lang_size: Dict[str, int]) -> Dict[str, int]:
    ratio_list = list(np.random.dirichlet(
        [alpha] * len(lang_size)) * whole_ratio)
    word_lang_dist = dict()
    for idx, i in enumerate(lang_size):
        word_lang_dist[i] = int(ratio_list[idx] * lang_size[i])
    return word_lang_dist


def calc_sim_matrix(images: torch.Tensor, dis_metric='cosine') -> torch.Tensor:
    n_img, D_img = images.size()
    if dis_metric == 'cosine':
        with torch.no_grad():
            dot = torch.tensordot(images, images, dims=([1], [1]))
            norm = torch.norm(images, dim=1)
            return dot / norm / norm.unsqueeze(1).expand_as(dot)
    else:
        raise NotImplementedError


def nearest_images(images: torch.Tensor, n: int):
    with torch.no_grad():
        sim_matrix = calc_sim_matrix(images)
        sim_matrix.fill_diagonal_(-1)
        _, indices = torch.topk(sim_matrix, n, dim=1)
    return indices
