import argparse
import os
import random
from typing import List
import copy

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from models import Listener, SpeakerListener
from utils import truncate_dicts, get_distractor_images


def load_model(args) -> nn.Module:
    model = SpeakerListener(args).to(device=args.device)
    model.speaker.load_state_dict(torch.load(os.path.join(args.save_dir,
                                                          'spk_params', f"pop{args.seed}.pt")))
    model.listener.load_state_dict(torch.load(os.path.join(args.save_dir,
                                                           'list_params', f"pop{args.seed}.pt")))
    model.beholder.load_state_dict(torch.load(os.path.join(args.save_dir,
                                                           'bhd_params', f"pop{args.seed}.pt")))
    return model


def generate_beams(spk: nn.Module, images: torch.Tensor, args) -> torch.Tensor:
    beam_list, beam_len_list = [], []
    # msg, msg_len = spk.bs(images[i * args.s2p_batch_size:
    #                                 min((i+1) * args.s2p_batch_size,
    #                                     images.size()[0])],
    #                         args.beam_size, args.seq_len, False)
    msg, msg_len = spk.bs(images,
                          args.beam_size, args.seq_len, False)
    beam_list.append(msg)
    beam_len_list.append(msg_len)
    return torch.cat(beam_list, dim=0), torch.cat(beam_len_list, dim=0)


def choose_captions_from_beams(listener: Listener, captions: torch.Tensor,
                               cap_len: torch.Tensor, image: torch.Tensor,
                               distractor_images: List[torch.Tensor]) \
        -> torch.Tensor:
    """Choose captions from beams with a listener.

    Args:
        listener: The listener model.        
        captions: The candidate captions (B, b, L).
        cap_len: Length of candidate captions (B, b).
        image: The target images (B, D_img).
        distractor_images: The distractor images (B, n_dis, D_img).

        B is the batch size, b being beam size, L sentence length, 
        D_img image embedding size, and n_dis being the number of distractors.

    Returns:
        indices of chosen captions

    """
    B, b, L = captions.size()
    _, D_img = image.size()
    with torch.no_grad():
        log_prob_list = []
        batch_size = B // b
        for i in range(0, B, batch_size):
            batch_slice = slice(i, min(i + batch_size, B))
            expanded_image = image[batch_slice].unsqueeze(1).expand(-1, b, -1)\
                .reshape(-1, D_img)
            expanded_distractor_images = [i[batch_slice]
                                          .unsqueeze(1).expand(-1, b, -1)
                                          .reshape(-1, D_img)
                                          for i in distractor_images]
            log_prob = - listener.get_loss_acc(expanded_image,
                                               expanded_distractor_images,
                                               captions[batch_slice].view(-1, L),
                                               cap_len[batch_slice].view(-1),
                                               reduction='none')[0]
            log_prob_list.append(log_prob)
    log_prob = torch.cat(log_prob_list, dim=0).view(B, b)
    return torch.max(log_prob, dim=-1)[1]


def get_caption_candidates(test_image, args):
    model = load_model(args)
    speaker_model = model.speaker
    test_caption_beams, test_cap_len = generate_beams(
        speaker_model, test_image, args)
    return test_caption_beams, test_cap_len


def same_partner_baseline(test_image, test_distractor_images,
                          test_caption_beams, test_cap_len, args):

    model = load_model(args)
    training_listener = model.listener
    chosen_indices = choose_captions_from_beams(training_listener,
                                                test_caption_beams,
                                                test_cap_len, test_image,
                                                test_distractor_images)
    test_caption = test_caption_beams[range(chosen_indices.size()[0]),
                                      chosen_indices]
    test_cap_len = test_cap_len[range(chosen_indices.size()[0]),
                                chosen_indices]
    _, acc = training_listener.get_loss_acc(test_image, test_distractor_images,
                                            test_caption, test_cap_len)
    return acc


def different_partner_baseline(test_image, test_distractor_images,
                               test_caption_beams, test_cap_len, args,
                               new_partner_args):
    model = load_model(args)
    training_listener = model.listener
    chosen_indices = choose_captions_from_beams(training_listener,
                                                test_caption_beams,
                                                test_cap_len, test_image,
                                                test_distractor_images)
    test_caption = test_caption_beams[range(chosen_indices.size()[0]),
                                      chosen_indices]
    test_cap_len = test_cap_len[range(chosen_indices.size()[0]),
                                chosen_indices]

    model = load_model(new_partner_args)
    listener = model.listener
    _, acc = listener.get_loss_acc(test_image, test_distractor_images,
                                   test_caption, test_cap_len)
    return acc


def overlap_with_training_listener(test_image, test_distractor_images,
                                   test_caption_beams, test_cap_len, args,
                                   new_partner_args):
    model = load_model(args)
    training_listener = model.listener
    chosen_indices = choose_captions_from_beams(training_listener,
                                                test_caption_beams,
                                                test_cap_len, test_image,
                                                test_distractor_images)
    test_caption = test_caption_beams[range(chosen_indices.size()[0]),
                                      chosen_indices]
    test_cap_len = test_cap_len[range(chosen_indices.size()[0]),
                                chosen_indices]

    _, _, preds = training_listener.get_loss_acc(test_image,
                                                 test_distractor_images,
                                                 test_caption, test_cap_len,
                                                 shuffle=False, output_pred=True)
    model = load_model(new_partner_args)
    listener = model.listener
    _, _, preds_new = listener.get_loss_acc(test_image, test_distractor_images,
                                            test_caption, test_cap_len, shuffle=False,
                                            output_pred=True)
    return np.mean(np.equal(preds, preds_new))


def different_partner_upperbound(test_image, test_distractor_images,
                                 test_caption_beams, test_cap_len, args,
                                 new_partner_args):
    model = load_model(new_partner_args)
    training_listener = model.listener
    chosen_indices = choose_captions_from_beams(training_listener,
                                                test_caption_beams,
                                                test_cap_len, test_image,
                                                test_distractor_images)
    test_caption = test_caption_beams[range(chosen_indices.size()[0]),
                                      chosen_indices]
    test_cap_len = test_cap_len[range(chosen_indices.size()[0]),
                                chosen_indices]

    model = load_model(new_partner_args)
    listener = model.listener
    _, acc = listener.get_loss_acc(test_image, test_distractor_images,
                                   test_caption, test_cap_len)
    return acc


def different_partner_upperbound_confidence(
        test_image, test_distractor_images, test_caption_beams, test_cap_len, args,
        new_partner_args):
    model = load_model(new_partner_args)
    training_listener = model.listener
    chosen_indices = choose_captions_from_beams(training_listener,
                                                test_caption_beams,
                                                test_cap_len, test_image,
                                                test_distractor_images)
    test_caption = test_caption_beams[range(chosen_indices.size()[0]),
                                      chosen_indices]
    test_cap_len = test_cap_len[range(chosen_indices.size()[0]),
                                chosen_indices]

    model = load_model(new_partner_args)
    listener = model.listener
    _, _, logits = listener.get_loss_acc(
        test_image, test_distractor_images, test_caption, test_cap_len,
        output_logits=True)
    confidence = torch.mean(torch.max(logits, dim=-1))
    return confidence


def different_partner_nonchosen_baseline(test_image, test_distractor_images,
                                         test_caption_beams, test_cap_len, args,
                                         new_partner_args):
    test_caption = test_caption_beams[:, 0]
    test_cap_len = test_cap_len[:, 0]

    model = load_model(new_partner_args)
    listener = model.listener
    _, acc = listener.get_loss_acc(test_image, test_distractor_images,
                                   test_caption, test_cap_len)
    return acc


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    print("OPTS:\n", vars(args))
    feat_path = args.coco_path
    data_path = args.coco_path

    (_, _, test_images) \
        = [torch.load('{}/feats/{}'.format(feat_path, x)) for x in
           "train_feats valid_feats test_feats".split()]
    test_images = test_images.to(device=args.device)

    (w2i, i2w) = [torch.load(data_path + 'dics/{}'.format(x))
                  for x in "w2i i2w".split()]
    w2i, i2w = truncate_dicts(w2i, i2w, args.num_words)
    args.vocab_size = len(w2i)
    args.w2i = w2i
    args.i2w = i2w

    test_caption_beams, test_cap_len = get_caption_candidates(
        test_images, args)

    distractors_images = get_distractor_images(test_images, args)
    print(
        f"Same partner baseline score: {same_partner_baseline(test_images, distractors_images, test_caption_beams, test_cap_len, args)}")

    new_args = copy.deepcopy(args)
    new_args.save_dir = new_args.listener_save_dir
    new_args.vocab_size = new_args.listener_vocab_size
    new_args.seed = new_args.listener_seed
    print(
        f"Different partner baseline score: {different_partner_baseline(test_images, distractors_images, test_caption_beams, test_cap_len, args, new_args)}")

    print(
        f"Different partner non-chosen baseline score: {different_partner_nonchosen_baseline(test_images, distractors_images, test_caption_beams, test_cap_len, args, new_args)}")

    print(
        f"Different partner upperbound score: {different_partner_upperbound(test_images, distractors_images, test_caption_beams, test_cap_len, args, new_args)}")

    print(
        f"Different partner overlap: {overlap_with_training_listener(test_images, distractors_images, test_caption_beams, test_cap_len, args, new_args)}")

    print(
        f"Different partner confidence score: {different_partner_upperbound_confidence(test_images, distractors_images, test_caption_beams, test_cap_len, args, new_args)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0,
                        help='seed')
    parser.add_argument("--num_seed_examples", type=int, default=1000,
                        help="Number of seed examples")
    parser.add_argument("--num_distrs", type=int, default=9,
                        help="Number of distractors")
    parser.add_argument("--s2p_schedule", type=str, default="sched",
                        help="s2p schedule")
    parser.add_argument("--s2p_selfplay_updates", type=int, default=50,
                        help="s2p self-play updates")
    parser.add_argument("--s2p_list_updates", type=int, default=50,
                        help="s2p listener supervised updates")
    parser.add_argument("--s2p_spk_updates", type=int, default=50,
                        help="s2p speaker supervised updates")
    parser.add_argument("--s2p_batch_size", type=int, default=1000,
                        help="s2p batch size")
    parser.add_argument("--pop_batch_size", type=int, default=1000,
                        help="Pop Batch size")
    parser.add_argument("--rand_perc", type=float, default=0.75,
                        help="rand perc")
    parser.add_argument("--sched_rand_frz", type=int, default=0.5,
                        help="sched_rand_frz perc")
    parser.add_argument("--num_words", type=int, default=100,
                        help="Number of words in the vocabulary")
    parser.add_argument("--seq_len", type=int, default=15,
                        help="Max Sequence length of speaker utterance")
    parser.add_argument("--unk_perc", type=float, default=0.3,
                        help="Max percentage of <UNK>")
    parser.add_argument("--max_iters", type=int, default=300,
                        help="max training iters")
    parser.add_argument("--D_img", type=int, default=2048,
                        help="ResNet feature dimensionality. Can't change this")
    parser.add_argument("--D_hid", type=int, default=512,
                        help="RNN hidden state dimensionality")
    parser.add_argument("--D_emb", type=int, default=256,
                        help="Token embedding (word) dimensionality")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout probability")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="Gumbel temperature")
    parser.add_argument("--hard", type=bool, default=True,
                        help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--min_list_steps", type=int, default=2000,
                        help="Min num of listener supervised steps")
    parser.add_argument("--min_spk_steps", type=int, default=1000,
                        help="Min num of speaker supervised steps")
    parser.add_argument("--test_every", type=int, default=10,
                        help="test interval")
    parser.add_argument("--seed_val_pct", type=float, default=0.1,
                        help="% of seed samples used as validation for early stopping")
    parser.add_argument('--coco_path', type=str, default="./coco/",
                        help="MSCOCO dir path")
    parser.add_argument("--save_dir", type=str, default="",
                        help="Save directory.")
    parser.add_argument("--sample_temp", type=float, default=30,
                        help="Temperature used for sampling difficult distractors")
    parser.add_argument("--sample_lang", action="store_true", default=False,
                        help="Sample the languges of the data")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Parameter of Dirichlet distribution")
    parser.add_argument("--beam_size", type=int, default=10,
                        help="Beam size")
    parser.add_argument("--listener_save_dir", type=str, default="",
                        help="Listener save dir")
    parser.add_argument("--listener_vocab_size", type=int, default=100,
                        help="Listener vocab size")
    parser.add_argument("--listener_seed", type=int, default=0,
                        help="Listener seed")

    args = parser.parse_args()
    main(args)
