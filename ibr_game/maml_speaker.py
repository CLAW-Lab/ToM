from typing import Tuple
import json
import argparse
import random
import copy
import os
import tqdm
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler

from models import Listener, AggregatedListener
from few_shot_learning_system import MAMLFewShotClassifier
from referential_game import get_caption_candidates, load_model
from utils import truncate_dicts, nearest_images
# from lang_id import language_identification

# torch.backends.cudnn.enabled = False

# torch.autograd.set_detect_anomaly(True)


def get_maml_args(maml_args_file: str):
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d
    return objectview(json.load(open(maml_args_file, "r")))


def sample_listener(listener_template, listener_range, args, idx=None) \
        -> Listener:
    if idx is None:
        listener_choice = random.choice(listener_range)
    else:
        listener_choice = listener_range[idx]
    new_args = copy.deepcopy(args)
    new_args.save_dir = listener_template.format(*listener_choice)
    new_args.vocab_size = new_args.listener_vocab_size
    new_args.seed = listener_choice[0]  # hardcode warning!!!
    return load_model(new_args).listener


def gen_game(n_img: int, N: int, M: int, n_distrs: int,
             sample_candidates: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    device = sample_candidates.device
    n = sample_candidates.size()[1]
    target_images = torch.randint(n_img, size=(N, M)).to(device)
    distr_images = torch.randint(n, size=(N, M, n_distrs + 1)).to(device)
    target_candidates = torch.index_select(
        sample_candidates, 0, target_images.view(-1)).view(N, M, n)
    distr_images = torch.gather(
        target_candidates, 2, distr_images).view(N*M, n_distrs+1)
    target_indices = torch.randint(n_distrs + 1, size=(N*M,)).to(device)
    distr_images[range(N*M), target_indices] = target_images.view(N*M)
    return distr_images, target_indices


def print_profile_of_tensor(x):
    print(x.size(), x.device)


def rollout(images: torch.Tensor, caption_beams: torch.Tensor,
            caption_beams_len: torch.Tensor, nearest_images: torch.Tensor,
            N: int, T: int, M: int, args, listener_range,
            maml_listener: MAMLFewShotClassifier, epoch_id: int) \
        -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    :param images: input images
    :param caption_beams: captions for the input images
    :param N: number of tasks
    :param T: number of interactions between speaker and listener
    :param M: number of target tasks (M-1 of them not used for training)
    """
    b = caption_beams.size()[1]
    L = caption_beams.size()[2]
    n_distrs = args.num_distrs
    D_img = images.size()[-1]
    n_img = images.size()[0]
    listeners = list()
    for i in range(N):
        while True:
            try:
                listeners.append(sample_listener(
                    args.listener_template, listener_range, args))
                break
            except FileNotFoundError:
                pass
    imgs_till_now = torch.zeros(N, 1, M, n_distrs + 1, D_img).to(args.device)
    caps_till_now = torch.zeros(N, 1, M, L).long().to(args.device)
    cap_lens_till_now = torch.ones(N, 1, M).long().to(args.device)
    ys_till_now = torch.zeros(N, 1, M).long().to(args.device)
    ys_logp_till_now = torch.zeros(N, 1, M, n_distrs + 1).to(args.device)

    rollout_stat = {}
    rollout_accuracy = []

    for i in range(T):
        imgs = imgs_till_now[:, :, 0]
        caps = caps_till_now[:, :, 0]
        cap_lens = cap_lens_till_now[:, :, 0]
        ys = ys_till_now[:, :, 0]
        ys_logp = ys_logp_till_now[:, :, 0]
        img_ids, gold_ys = gen_game(n_img, N, M, n_distrs, nearest_images)
        img_ids, gold_ys = img_ids.to(args.device), gold_ys.to(args.device)
        img_tgt_ids = img_ids[range(N * M), gold_ys]
        cap_candidates = torch.index_select(caption_beams, 0, img_tgt_ids)
        cap_len_candidates = torch.index_select(
            caption_beams_len, 0, img_tgt_ids)
        img_ids = img_ids.unsqueeze(1).expand(-1, b, -1)
        gold_ys = gold_ys.unsqueeze(1).expand(-1, b)
        img_ids = img_ids.reshape(-1)
        imgs_new = torch.index_select(
            images, 0, img_ids).view(N*M, b, n_distrs+1, D_img)
        img_ids = img_ids.view(N*M, b, n_distrs+1)

        imgs_new = imgs_new.view(N, M * b, n_distrs+1, D_img)
        cap_candidates = cap_candidates.view(N, M*b, L)
        cap_len_candidates = cap_len_candidates.view(N, M*b)

        x_support = (imgs, caps, cap_lens)
        y_support = ys if not args.maml_args.soft_y_support else ys_logp
        x_target = (imgs_new, cap_candidates, cap_len_candidates)
        y_target = torch.zeros(N, M*b).long().to(args.device)  # dummy y_target
        _, preds = maml_listener.run_validation_iter(
            (x_support, x_target, y_support, y_target))
        preds = torch.stack(preds, dim=0)
        preds = F.log_softmax(preds, dim=-1)
        preds_acc = torch.gather(
            preds, -1, gold_ys.reshape(N, M * b, 1)).view(N, M, b)
        caption_choice = torch.max(preds_acc, -1)[1].view(N * M)
        cap_candidates = cap_candidates.view(N*M, b, L)
        cap_new = cap_candidates[range(N * M), caption_choice]
        cap_len_candidates = cap_len_candidates.view(N*M, b)
        cap_len_new = cap_len_candidates[range(N * M), caption_choice]

        imgs_new = imgs_new.view(N, M, b, n_distrs+1, D_img)
        imgs_new = imgs_new[:, :, 0]
        cap_new = cap_new.view(N, M, L)
        cap_len_new = cap_len_new.view(N, M)

        y, y_logp = [], []
        with torch.no_grad():
            for j in range(N):
                preds_out, preds_logp = \
                    listeners[j].predict(
                        imgs_new[j].view(M, n_distrs+1, D_img),
                        cap_new[j].view(M, L),
                        cap_len_new[j].view(M),
                        output_logp=True)
                y.append(preds_out)
                y_logp.append(preds_logp)
        y = torch.stack(y, dim=0)
        y_logp = torch.stack(y_logp, dim=0)
        rollout_accuracy.append(gold_ys[:, 0] == y.view(-1))

        imgs_till_now = torch.cat(
            [imgs_till_now, imgs_new.unsqueeze(1)], dim=1)
        caps_till_now = torch.cat([caps_till_now, cap_new.unsqueeze(1)], dim=1)
        cap_lens_till_now = torch.cat(
            [cap_lens_till_now, cap_len_new.unsqueeze(1)], dim=1)
        ys_till_now = torch.cat([ys_till_now, y.unsqueeze(1)], dim=1)
        ys_logp_till_now = torch.cat([ys_logp_till_now, y_logp.unsqueeze(1)],
                                     dim=1)

    rollout_accuracy = torch.cat(rollout_accuracy, dim=0)
    rollout_stat["acc"] = torch.mean(rollout_accuracy.float())

    return (imgs_till_now, caps_till_now, cap_lens_till_now), ys_till_now, \
        ys_logp_till_now, rollout_stat


def rollout_case_study(
    images: torch.Tensor, caption_beams: torch.Tensor,
    caption_beams_len: torch.Tensor, nearest_images: torch.Tensor,
    N: int, T: int, M: int, args, listener_range,
    maml_listener: MAMLFewShotClassifier, epoch_id: int) \
        -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    :param images: input images
    :param caption_beams: captions for the input images
    :param N: number of tasks
    :param T: number of interactions between speaker and listener
    :param M: number of target tasks (M-1 of them not used for training)
    """
    b = caption_beams.size()[1]
    L = caption_beams.size()[2]
    n_distrs = args.num_distrs
    D_img = images.size()[-1]
    n_img = images.size()[0]
    listeners = list()
    for i in range(N):
        try:
            listeners.append(sample_listener(
                args.listener_template, listener_range, args, i))
        except:
            pass
    N = len(listeners)
    imgs_till_now = torch.zeros(N, 1, M, n_distrs + 1, D_img).to(args.device)
    caps_till_now = torch.zeros(N, 1, M, L).long().to(args.device)
    cap_lens_till_now = torch.ones(N, 1, M).long().to(args.device)
    ys_till_now = torch.zeros(N, 1, M).long().to(args.device)
    ys_gold_till_now = torch.zeros(N, 1, M).long().to(args.device)
    ys_logp_till_now = torch.zeros(N, 1, M, n_distrs + 1).to(args.device)

    rollout_stat = {}
    rollout_accuracy = []

    for i in range(T):
        imgs = imgs_till_now[:, :, 0]
        caps = caps_till_now[:, :, 0]
        cap_lens = cap_lens_till_now[:, :, 0]
        ys = ys_till_now[:, :, 0]
        ys_gold = ys_gold_till_now[:, :, 0]
        ys_logp = ys_logp_till_now[:, :, 0]
        img_ids, gold_ys = gen_game(n_img, 1, M, n_distrs, nearest_images)
        img_ids = img_ids.repeat(N, 1)
        gold_ys = gold_ys.repeat(N)

        img_ids, gold_ys = img_ids.to(args.device), gold_ys.to(args.device)
        img_tgt_ids = img_ids[range(N * M), gold_ys]
        cap_candidates = torch.index_select(caption_beams, 0, img_tgt_ids)
        cap_len_candidates = torch.index_select(
            caption_beams_len, 0, img_tgt_ids)
        img_ids = img_ids.unsqueeze(1).expand(-1, b, -1)
        gold_ys = gold_ys.unsqueeze(1).expand(-1, b)
        img_ids = img_ids.reshape(-1)
        imgs_new = torch.index_select(
            images, 0, img_ids).view(N*M, b, n_distrs+1, D_img)
        img_ids = img_ids.view(N*M, b, n_distrs+1)

        imgs_new = imgs_new.view(N, M * b, n_distrs+1, D_img)
        cap_candidates = cap_candidates.view(N, M*b, L)
        cap_len_candidates = cap_len_candidates.view(N, M*b)

        x_support = (imgs, caps, cap_lens)
        y_support = ys if not args.maml_args.soft_y_support else ys_logp
        x_target = (imgs_new, cap_candidates, cap_len_candidates)
        y_target = torch.zeros(N, M*b).long().to(args.device)  # dummy y_target
        if args.binary_loss:
            y_support = (y_support, ys_gold)
            y_target = (y_target, torch.zeros(N, M*b).long().to(args.device))
        _, preds = maml_listener.run_validation_iter(
            (x_support, x_target, y_support, y_target))
        preds = torch.stack(preds, dim=0)
        preds = F.log_softmax(preds, dim=-1)
        preds_acc = torch.gather(
            preds, -1, gold_ys.reshape(N, M * b, 1)).view(N, M, b)
        caption_choice = torch.max(preds_acc, -1)[1].view(N * M)
        cap_candidates = cap_candidates.view(N*M, b, L)
        cap_new = cap_candidates[range(N * M), caption_choice]
        cap_len_candidates = cap_len_candidates.view(N*M, b)
        cap_len_new = cap_len_candidates[range(N * M), caption_choice]

        imgs_new = imgs_new.view(N, M, b, n_distrs+1, D_img)
        imgs_new = imgs_new[:, :, 0]
        cap_new = cap_new.view(N, M, L)
        cap_len_new = cap_len_new.view(N, M)

        y, y_logp = [], []
        with torch.no_grad():
            for j in range(N):
                preds_out, preds_logp = \
                    listeners[j].predict(
                        imgs_new[j].view(M, n_distrs+1, D_img),
                        cap_new[j].view(M, L),
                        cap_len_new[j].view(M),
                        output_logp=True)
                y.append(preds_out)
                y_logp.append(preds_logp)
        y = torch.stack(y, dim=0)
        y_logp = torch.stack(y_logp, dim=0)
        rollout_accuracy.append(gold_ys[:, 0] == y.view(-1))

        imgs_till_now = torch.cat(
            [imgs_till_now, imgs_new.unsqueeze(1)], dim=1)
        caps_till_now = torch.cat([caps_till_now, cap_new.unsqueeze(1)], dim=1)
        cap_lens_till_now = torch.cat(
            [cap_lens_till_now, cap_len_new.unsqueeze(1)], dim=1)
        ys_till_now = torch.cat([ys_till_now, y.unsqueeze(1)], dim=1)
        # ys_gold_till_now = torch.cat([ys_gold_till_now, gold_ys.unsqueeze(1)], dim=1)
        ys_logp_till_now = torch.cat([ys_logp_till_now, y_logp.unsqueeze(1)],
                                     dim=1)

    rollout_stat["per_step_acc"] \
        = [torch.mean(i.float()).item() for i in rollout_accuracy]
    rollout_accuracy = torch.cat(rollout_accuracy, dim=0)
    rollout_stat["acc"] = torch.mean(rollout_accuracy.float()).item()

    return (imgs_till_now, caps_till_now, cap_lens_till_now), ys_till_now, \
        ys_logp_till_now, rollout_stat


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    print("OPTS:\n", vars(args))
    feat_path = args.coco_path
    data_path = args.coco_path

    train_images, val_images, test_images \
        = [torch.load('{}/feats/{}'.format(feat_path, x)) for x in
           "train_feats valid_feats test_feats".split()]
    train_images = train_images.to(device=args.device)
    val_images = val_images.to(device=args.device)
    test_images = test_images.to(device=args.device)

    if args.image_sample_ratio == 1.0:
        train_nearest_images = torch.arange(
            train_images.size()[0]).expand(train_images.size()[0], -1)
        val_nearest_images = torch.arange(
            val_images.size()[0]).expand(val_images.size()[0], -1)
        test_nearest_images = torch.arange(
            test_images.size()[0]).expand(test_images.size()[0], -1)
    else:
        train_nearest_images = nearest_images(train_images,
                                              n=int(args.image_sample_ratio * train_images.size()[0]))
        torch.cuda.empty_cache()
        val_nearest_images = nearest_images(val_images, n=int(
            args.image_sample_ratio * val_images.size()[0]))
        test_nearest_images = nearest_images(test_images, n=int(
            args.image_sample_ratio * test_images.size()[0]))

    (w2i, i2w) = [torch.load(data_path + 'dics/{}'.format(x))
                  for x in "w2i i2w".split()]
    w2i, i2w = truncate_dicts(w2i, i2w, args.num_words)
    args.vocab_size = len(w2i)
    args.w2i = w2i
    args.i2w = i2w

    if os.path.exists(f"save/train_caption_beams_{args.beam_size}.pt") \
            and os.path.exists(f"save/train_cap_len_{args.beam_size}.pt"):
        train_caption_beams = torch.load(
            f"save/train_caption_beams_{args.beam_size}.pt").to(args.device)
        train_cap_len = torch.load(f"save/train_cap_len_{args.beam_size}.pt")\
            .to(args.device)
    else:
        train_caption_beams, train_cap_len = get_caption_candidates(
            train_images, args)
        torch.save(train_caption_beams,
                   f"save/train_caption_beams_{args.beam_size}.pt")
        torch.save(train_cap_len, f"save/train_cap_len_{args.beam_size}.pt")

    if os.path.exists(f"save/val_caption_beams_{args.beam_size}.pt") \
            and os.path.exists(f"save/val_cap_len_{args.beam_size}.pt"):
        val_caption_beams = torch.load(
            f"save/val_caption_beams_{args.beam_size}.pt").to(args.device)
        val_cap_len = torch.load(f"save/val_cap_len_{args.beam_size}.pt")\
            .to(args.device)
    else:
        val_caption_beams, val_cap_len = get_caption_candidates(
            val_images, args)
        torch.save(val_caption_beams,
                   f"save/val_caption_beams_{args.beam_size}.pt")
        torch.save(val_cap_len, f"save/val_cap_len_{args.beam_size}.pt")

    args.maml_args = get_maml_args(args.maml_args)
    if args.use_mono_listeners:
        new_args = copy.deepcopy(args)
        new_args.vocab_size = new_args.listener_vocab_size
        new_args.D_hid = new_args.D_hid_maml_listener
        maml_listener = MAMLFewShotClassifier(
            AggregatedListener, args.maml_args, new_args
        )
        for idx, i in enumerate(maml_listener.classifier.listeners):
            i.load_state_dict(
                torch.load(os.path.join(f"save/mono_student_{idx}",
                                        'list_params', f"pop{idx}.pt"))
            )
    else:
        new_args = copy.deepcopy(args)
        new_args.D_hid = new_args.D_hid_maml_listener
        maml_listener = MAMLFewShotClassifier(
            Listener, args.maml_args, new_args
        )
        maml_listener.classifier.load_state_dict(
            torch.load(os.path.join(args.save_dir,
                                    'list_params', f"pop{args.seed}.pt"))
        )

    if args.fully_offline:
        maml_listener_freeze = copy.deepcopy(maml_listener)
        for i in maml_listener_freeze.parameters():
            i.requires_grad_ = False

    max_acc = 0.0

    if args.evaluate_model_filename != "":
        maml_listener.load_state_dict(torch.load(args.evaluate_model_filename))
        if os.path.exists(f"save/test_caption_beams_{args.beam_size}.pt") \
                and os.path.exists(f"save/test_cap_len_{args.beam_size}.pt"):
            test_caption_beams = torch.load(
                f"save/test_caption_beams_{args.beam_size}.pt").to(args.device)
            test_cap_len = torch.load(f"save/test_cap_len_{args.beam_size}.pt")\
                .to(args.device)
        else:
            test_caption_beams, test_cap_len = get_caption_candidates(
                test_images, args)
            torch.save(test_caption_beams,
                       f"save/test_caption_beams_{args.beam_size}.pt")
            torch.save(test_cap_len, f"save/test_cap_len_{args.beam_size}.pt")
        average_stat = {'acc': [], 'per_step_acc': [[]
                                                    for _ in range(args.maml_time_step)]}
        for i in range(1):
            (imgs, caps, cap_lens), ys, ys_logp, rollout_stat = \
                rollout_case_study(test_images, test_caption_beams, test_cap_len,
                                   test_nearest_images,
                                   20, args.maml_time_step, 50, args,
                                   [(i,) for i in range(80, 100)], maml_listener, 0)
            average_stat['acc'].append(rollout_stat['acc'])
            for j, k in zip(average_stat['per_step_acc'], rollout_stat['per_step_acc']):
                j.append(k)
        average_stat['acc'] = sum(average_stat['acc']) / \
            len(average_stat['acc'])
        for i in range(args.maml_time_step):
            average_stat['per_step_acc'][i] = sum(average_stat['per_step_acc'][i]) / \
                len(average_stat['per_step_acc'][i])
        print(average_stat)
        # for i in range(100):
        #     (imgs, caps, cap_lens), ys, ys_logp, rollout_stat = \
        #         rollout_case_study(test_images, test_caption_beams, test_cap_len,
        #                         test_nearest_images,
        #                         20, args.maml_time_step, 1, args,
        #                         [(i,) for i in range(80, 100)], maml_listener, 0)

        #     data = []
        #     for i in range(caps.size()[0]):
        #         lang_list = []
        #         for j in range(args.maml_time_step):
        #             sent = caps[i][j][0][:cap_lens[i][j][0]].tolist()
        #             lang = language_identification(sent)
        #             lang_list.append(lang)
        #         data.append(lang_list)

        #     last_lang = set()
        #     for lang_list in data:
        #         last_lang.add(lang_list[-1])
        #     if len(last_lang) > 3:
        #         for i in range(caps.size()[0]):
        #             line = [str(i)]
        #             for j in range(args.maml_time_step):
        #                 line.append(
        #                     ' '.join(
        #                         map(lambda x: args.i2w[x.item()], caps[i][j][0][:cap_lens[i][j][0]]))
        #                 )
        #                 for k in range(args.num_distrs+1):
        #                     line.append(str(torch.exp(ys_logp[i][j][0][k]).item()))
        #             print('\t'.join(line))
        exit()

    for epoch in range(args.maml_args.total_epochs):
        mean_losses = {}
        pbar = tqdm.tqdm(
            range(args.maml_args.total_iter_per_epoch), dynamic_ncols=True)
        for _ in pbar:
            # Rollout Phase: collecting data from current maml model
            rollout_start_time = time.time()
            (imgs, caps, cap_lens), ys, ys_logp, rollout_stat \
                = rollout(
                train_images, train_caption_beams, train_cap_len,
                train_nearest_images,
                args.maml_args.batch_size, args.maml_time_step,
                args.maml_args.parallel_target_tasks, args,
                [(i,) for i in range(80)],
                maml_listener_freeze if args.fully_offline else maml_listener,
                epoch)
            rollout_acc = rollout_stat["acc"]
            if "rollout_acc" not in mean_losses:
                mean_losses["rollout_acc"] = [rollout_acc]
            else:
                mean_losses["rollout_acc"].append(rollout_acc)
            rollout_end_time = time.time()
            # MAML Phase: train maml model on the collected data
            maml_start_time = time.time()
            for t in range(1, args.maml_time_step+1):
                x_support = (imgs[:, :t, 0], caps[:, :t, 0],
                             cap_lens[:, :t, 0])
                y_support = ys[:, :t, 0]
                x_target = (imgs[:, t], caps[:, t], cap_lens[:, t])
                # y_target = ys[:, t:t+1]
                y_target = ys_logp[:, t]
                losses, preds = maml_listener.run_train_iter(
                    (x_support, x_target, y_support, y_target), epoch)
                preds = torch.stack(preds, dim=0)
                preds = F.log_softmax(preds, dim=-1)
                for loss in losses:
                    if loss not in mean_losses:
                        mean_losses[loss] = [losses[loss]]
                    else:
                        mean_losses[loss].append(losses[loss])
            maml_end_time = time.time()
            pbar.set_postfix({'rollout time': rollout_end_time - rollout_start_time,
                              'maml time': maml_end_time - maml_start_time})
        for i in mean_losses:
            mean_losses[i] = sum(mean_losses[i]) / len(mean_losses[i])
        print(mean_losses)
        _, _, _, rollout_stat = rollout(
            val_images, val_caption_beams, val_cap_len,
            val_nearest_images,
            100, args.maml_time_step, 10, args,
            [(i,) for i in range(80, 100)], maml_listener, epoch)
        print(f"Accuracy on the val set: {rollout_stat['acc']}")
        if rollout_stat['acc'] > max_acc and args.maml_save_dir != "":
            max_acc = rollout_stat['acc']
            Path(args.maml_save_dir).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(
                args.maml_save_dir, f"maml_listener_{max_acc}.pt")
            print(f"Saving model to {save_path}")
            torch.save(maml_listener.state_dict(), save_path)


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
    parser.add_argument("--D_hid_maml_listener", type=int, default=512,
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
    parser.add_argument("--maml_args", type=str, default="",
                        help="MAML arugments")
    parser.add_argument("--listener_template", type=str, default="",
                        help="listener's save dir template")
    parser.add_argument("--maml_time_step", type=int, default=20)
    parser.add_argument("--maml_save_dir", type=str, default="")
    parser.add_argument("--evaluate_model_filename", type=str, default="")
    parser.add_argument("--use_mono_listeners", action="store_true")
    parser.add_argument("--image_sample_ratio", type=float, default=1.0)
    parser.add_argument("--fully_offline", action="store_true", default=False,
                        help="freeze the listener in rollout")
    parser.add_argument("--binary_loss", action="store_true", default=False,
                        help="Using binary loss")
    args = parser.parse_args()
    main(args)
