import numpy as np
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import utils as U

from beam_search import beam_search


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        name = name.replace("layer_dict.", "")
        name = name.replace("block_dict.", "")
        name = name.replace("module-", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key,
                        value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    # print(current_dict.keys(), output_dict.keys())
    return output_dict


class Beholder(nn.Module):
    def __init__(self, args):
        super(Beholder, self).__init__()
        self.img_to_hid = nn.Linear(args.D_img, args.D_hid)
        self.drop = nn.Dropout(p=args.dropout)

    def forward(self, img):
        h_img = img
        h_img = self.img_to_hid(h_img)
        h_img = self.drop(h_img)
        return h_img


class AggregatedListener(nn.Module):
    def __init__(self, args):
        super(AggregatedListener, self).__init__()
        self.listeners = nn.ModuleList(
            [Listener(args) for _ in range(10)])
        for params in self.parameters():
            params.requires_grad = False
        self.log_psi = nn.Parameter(torch.zeros(10))

    def maml_forward(self, x, params, training, backup_running_statistics, num_step):
        for name, param in params.items():
            submodule_names, parameter_name = name.split(
                '.')[:-1], name.split('.')[-1]
            current_module = self
            for i in submodule_names:
                current_module = current_module.__getattr__(i)
            object.__setattr__(current_module, parameter_name, param[0])
            if len(submodule_names) and submodule_names[0] == "rnn":
                self.rnn._flat_weights = [
                    (lambda wn: getattr(self.rnn, wn) if hasattr(
                        self.rnn, wn) else None)(wn)
                    for wn in self.rnn._flat_weights_names]
                self.rnn.flatten_parameters()

        with torch.no_grad():
            logits_list = [self.listeners[i].maml_forward(
                x) for i in range(10)]
        logits = torch.stack(logits_list)
        logits = torch.log_softmax(logits, dim=-1)
        log_psi = torch.log_softmax(self.log_psi, dim=0).unsqueeze(
            1).unsqueeze(2).expand_as(logits)
        return torch.logsumexp(log_psi + logits, dim=0)

    def sim(self, x, params, y):
        for name, param in params.items():
            submodule_names, parameter_name = name.split(
                '.')[:-1], name.split('.')[-1]
            current_module = self
            for i in submodule_names:
                current_module = current_module.__getattr__(i)
            object.__setattr__(current_module, parameter_name, param[0])
            if len(submodule_names) and submodule_names[0] == "rnn":
                self.rnn._flat_weights = [
                    (lambda wn: getattr(self.rnn, wn) if hasattr(
                        self.rnn, wn) else None)(wn)
                    for wn in self.rnn._flat_weights_names]
                self.rnn.flatten_parameters()

        with torch.no_grad():
            logits_list = [self.listeners[i].maml_forward(
                x) for i in range(10)]
        logits = torch.stack(logits_list)
        logits = torch.softmax(logits, dim=-1)
        try:
            y_ = torch.softmax(y, dim=-1).unsqueeze(0).expand_as(logits)
            nearest = torch.argmin(
                torch.sum(torch.abs(logits - y_), dim=-1), dim=0)
            import ipdb
            ipdb.set_trace()
        except:
            pass

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        pass


class Listener(nn.Module):

    def __init__(self, args, beholder=None):
        super(Listener, self).__init__()
        self.rnn = nn.GRU(args.D_emb, args.D_hid, 1, batch_first=True)
        self.emb = nn.Linear(args.vocab_size, args.D_emb)
        self.hid_to_hid = nn.Linear(args.D_hid, args.D_hid)
        self.drop = nn.Dropout(p=args.dropout)
        self.D_hid = args.D_hid
        self.D_emb = args.D_emb
        self.vocab_size = args.vocab_size
        self.i2w = args.i2w
        self.w2i = args.w2i
        if beholder is None:
            self.beholder = Beholder(args)
        else:
            self.beholder = beholder
        # self.loss_fn = nn.CrossEntropyLoss(reduction='none').to(device=args.device)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, spk_msg, spk_msg_lens):
        batch_size = spk_msg.shape[0]

        h_0 = torch.zeros(1, batch_size, self.D_hid, device=device)

        if spk_msg.type() in ['torch.FloatTensor', 'torch.cuda.FloatTensor']:
            spk_msg_emb = self.emb(spk_msg.float())
        elif spk_msg.type() in ['torch.LongTensor', 'torch.cuda.LongTensor']:
            spk_msg[spk_msg > self.vocab_size] = self.w2i["<UNK>"]
            spk_msg_emb = F.embedding(
                spk_msg.clone(), self.emb.weight.transpose(0, 1))
            spk_msg_emb += self.emb.bias
        else:
            print(spk_msg.type())
            raise NotImplementedError
        spk_msg_emb = self.drop(spk_msg_emb)

        try:
            pack = nn.utils.rnn.pack_padded_sequence(
                spk_msg_emb, spk_msg_lens, batch_first=True, enforce_sorted=False)
        except:
            import pdb
            pdb.set_trace()

        self.rnn.flatten_parameters()
        _, h_n = self.rnn(pack, h_0)
        h_n = h_n[-1:, :, :]
        out = h_n.transpose(0, 1).view(batch_size, self.D_hid)
        out = self.hid_to_hid(out)
        return out

    def maml_forward(self, x, params={}, training=None,
                     backup_running_statistics=None, num_step=None):
        imgs, caps, cap_lens = x

        for name, param in params.items():
            submodule_names, parameter_name = name.split(
                '.')[:-1], name.split('.')[-1]
            current_module = self
            for i in submodule_names:
                current_module = current_module.__getattr__(i)
            object.__setattr__(current_module, parameter_name, param[0])
            if submodule_names[0] == "rnn":
                self.rnn._flat_weights = [
                    (lambda wn: getattr(self.rnn, wn) if hasattr(
                        self.rnn, wn) else None)(wn)
                    for wn in self.rnn._flat_weights_names]
                self.rnn.flatten_parameters()

        h_pred = self.forward(caps, cap_lens.cpu())
        h_pred = h_pred.unsqueeze(1).repeat(1, imgs.size()[1], 1)

        h_img = self.beholder(imgs)

        logits = 1 / torch.mean(torch.pow(h_pred - h_img, 2),
                                2).view(-1, imgs.size()[1])

        return logits

    def get_loss_acc(self, image, distractor_images, spk_msg, spk_msg_lens,
                     reduction='mean', shuffle=True, output_pred=False,
                     output_logits=False):
        batch_size = spk_msg.shape[0]

        if reduction != 'none':
            spk_msg_lens, sorted_indices = torch.sort(
                spk_msg_lens, descending=True)
            spk_msg = spk_msg.index_select(0, sorted_indices)
            image = image.index_select(0, sorted_indices)

        h_pred = self.forward(spk_msg, spk_msg_lens.cpu())
        h_pred = h_pred.unsqueeze(1).repeat(1, 1 + len(distractor_images), 1)

        all_images = len(distractor_images) + 1
        img_idx = [list(range(all_images)) for _ in range(batch_size)]
        for c in img_idx:
            if shuffle:
                random.shuffle(c)

        target_idx = torch.tensor(
            np.argmax(np.array(img_idx) == 0, -1), dtype=torch.long, device=device)

        h_img = [self.beholder(image)] + [self.beholder(img)
                                          for img in distractor_images]
        h_img = torch.stack(h_img, dim=0).permute(1, 0, 2)
        for i in range(batch_size):
            h_img[i] = h_img[i, img_idx[i], :]

        logits = 1 / torch.mean(torch.pow(h_pred - h_img, 2),
                                2).view(-1, 1 + len(distractor_images))

        pred_outs = torch.argmax(logits, dim=-1).cpu().numpy()
        batch_inds = target_idx.cpu().numpy()

        acc = np.mean(np.equal(batch_inds, pred_outs))
        loss = F.cross_entropy(logits, target_idx, reduction=reduction)
        if not output_pred:
            if not output_logits:
                return loss, acc
            else:
                return loss, acc, logits
        else:
            if not output_logits:
                return loss, acc, pred_outs
            else:
                return loss, acc, pred_outs, logits

    def predict(self, images, spk_msg, spk_msg_lens, output_logp=False):
        h_pred = self.forward(spk_msg, spk_msg_lens.cpu())
        h_pred = h_pred.unsqueeze(1).repeat(1, images.size()[1], 1)

        h_img = self.beholder(images)

        logits = 1 / torch.mean(torch.pow(h_pred - h_img, 2),
                                2).view(-1, images.size()[1])

        pred_outs = torch.argmax(logits, dim=-1)
        if output_logp:
            return pred_outs, torch.log_softmax(logits, dim=-1)
        else:
            return pred_outs

    def test(self, image, distractor_images, spk_msg, spk_msg_lens):
        self.eval()
        loss, acc = self.get_loss_acc(
            image, distractor_images, spk_msg, spk_msg_lens)
        return loss.detach().cpu().numpy(), acc

    def update(self, image, distractor_images, spk_msg, spk_msg_lens):
        self.train()
        loss, acc = self.get_loss_acc(
            image, distractor_images, spk_msg, spk_msg_lens)
        self.optimizer.zero_grad()
        loss.backward()
        return loss, acc

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        pass


class Speaker(nn.Module):

    def __init__(self, args, beholder):
        super(Speaker, self).__init__()
        self.rnn = nn.GRU(args.D_emb, args.D_hid, 1, batch_first=True)
        self.emb = nn.Embedding(args.vocab_size, args.D_emb, padding_idx=0)
        self.hid_to_voc = nn.Linear(args.D_hid, args.vocab_size)
        self.D_emb = args.D_emb
        self.D_hid = args.D_hid
        self.drop = nn.Dropout(p=args.dropout)
        self.vocab_size = args.vocab_size
        self.i2w = args.i2w
        self.w2i = args.w2i
        self.temp = args.temp
        self.hard = args.hard
        self.seq_len = args.seq_len
        self.beholder = beholder
        self.loss_fn = nn.CrossEntropyLoss(reduce=False)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, image):
        batch_size, gen_idx, done, hid, input, msg_lens \
            = self.prepare_input(image)

        for idx in range(self.seq_len):
            # input = F.relu(input)
            output, hid = self.rnn_step(hid, input)

            top1, topi = U.gumbel_softmax(output, self.temp, self.hard)
            gen_idx.append(top1)

            for ii in range(batch_size):
                if topi[ii] == self.w2i["<EOS>"]:
                    done[ii] = True
                    msg_lens[ii] = idx + 1
                if np.array_equal(done, np.array([True for _ in range(batch_size)])):
                    break

            input = self.emb(topi)

        gen_idx = torch.stack(gen_idx).permute(1, 0, 2)
        msg_lens = torch.tensor(msg_lens, dtype=torch.long, device=device)
        return gen_idx, msg_lens

    def rnn_step(self, hid, input):
        self.rnn.flatten_parameters()
        output, hid = self.rnn(input, hid)
        output = output.view(-1, self.D_hid)
        output = self.hid_to_voc(output)
        output = output.view(-1, self.vocab_size)
        return output, hid

    def prepare_input(self, image):
        batch_size = image.shape[0]
        h_img = self.beholder(image).detach()
        start = [self.w2i["<BOS>"] for _ in range(batch_size)]
        gen_idx = []
        done = np.array([False for _ in range(batch_size)])
        h_img = h_img.unsqueeze(0).view(1, -1, self.D_hid).repeat(1, 1, 1)
        hid = h_img
        ft = torch.tensor(start, dtype=torch.long,
                          device=device).view(-1).unsqueeze(1)
        input = self.emb(ft)
        msg_lens = [self.seq_len for _ in range(batch_size)]
        return batch_size, gen_idx, done, hid, input, msg_lens

    def step(self, hid, inputs):
        self.rnn.flatten_parameters()
        # output, hid = self.rnn(
        #     F.relu(self.emb(inputs.unsqueeze(1))), hid.unsqueeze(0))
        output, hid = self.rnn(
            self.emb(inputs.unsqueeze(1)), hid.unsqueeze(0))

        output = output.view(-1, self.D_hid)
        output = self.hid_to_voc(output)
        output = F.log_softmax(output.view(-1, self.vocab_size))

        return output, hid

    def batchify(self, image, batch_size):
        h_img = self.beholder(image).detach().view(-1, self.D_hid)
        for i in range(0, image.shape[0], batch_size):
            yield h_img[i: i + batch_size]

    def bs(self, image, beam_size, max_len, choose_max=True):
        batch_size = image.shape[0]

        with torch.no_grad():
            generated_sents, _ = \
                beam_search(self.step, beam_size, max_len,
                            self.w2i["<EOS>"], self.w2i["<BOS>"],
                            self.batchify(image, batch_size//beam_size))

        if choose_max:
            generated_sents = generated_sents[:, 0, :]
            msg_lens = [max_len for i in range(batch_size)]
        else:
            generated_sents = generated_sents.view(batch_size * beam_size, -1)
            msg_lens = [max_len for i in range(batch_size * beam_size)]

        print("Beams generated, now calculating lengths")
        generated_sents_cpu = generated_sents.cpu()
        for idx, i in enumerate(generated_sents_cpu):
            for jdx, j in enumerate(i):
                if self.w2i["<EOS>"] == j:
                    msg_lens[idx] = jdx
                    break

        msg_lens = torch.tensor(msg_lens, dtype=torch.long,
                                device=generated_sents.get_device())

        if not choose_max:
            generated_sents = generated_sents.view(batch_size, beam_size, -1)
            msg_lens = msg_lens.view(batch_size, beam_size)

        return generated_sents, msg_lens

    def get_loss(self, image, caps, caps_lens, word_loss=False):
        batch_size = caps.shape[0]
        mask = (torch.arange(self.seq_len, device=device).expand(batch_size, self.seq_len) < caps_lens.unsqueeze(
            1)).float()

        caps_in = caps[:, :-1]
        caps_out = caps[:, 1:]

        h_img = self.beholder(image).detach()
        h_img = h_img.view(1, batch_size, self.D_hid).repeat(1, 1, 1)

        caps_in_emb = self.emb(caps_in)
        caps_in_emb = self.drop(caps_in_emb)

        self.rnn.flatten_parameters()
        output, _ = self.rnn(caps_in_emb, h_img)
        logits = self.hid_to_voc(output)

        loss = 0
        for j in range(logits.size(1)):
            flat_score = logits[:, j, :]
            flat_mask = mask[:, j]
            flat_tgt = caps_out[:, j]
            nll = self.loss_fn(flat_score, flat_tgt)
            loss += (flat_mask * nll).sum()

        if word_loss:
            loss /= mask.sum()
        return loss

    def test(self, image, caps, caps_lens):
        self.eval()
        loss = self.get_loss(image, caps, caps_lens)
        return loss.detach().cpu().numpy()

    def update(self, image, caps, caps_lens):
        self.train()
        loss = self.get_loss(image, caps, caps_lens)
        self.optimizer.zero_grad()
        loss.backward()
        return loss


class ReinforceSpeaker(Speaker):
    def forward(self, image):
        batch_size, gen_idx, done, hid, input, msg_lens \
            = self.prepare_input(image)

        running_logprob = torch.zeros(batch_size)
        for idx in range(self.seq_len):
            # input = F.relu(input)
            output, hid = self.rnn_step(hid, input)

            next_token, log_prob = U.sample(output, self.temp)
            gen_idx.append(next_token)

            for ii in range(batch_size):
                if next_token[ii] == self.w2i["<EOS>"]:
                    done[ii] = True
                    msg_lens[ii] = idx + 1
                else:
                    running_logprob[ii] += log_prob[ii]
                if np.array_equal(done,
                                  np.array([True for _ in range(batch_size)])):
                    break

            input = self.emb(next_token)

        gen_idx = torch.stack(gen_idx).permute(1, 0, 2)
        msg_lens = torch.tensor(msg_lens, dtype=torch.long, device=device)
        return gen_idx, msg_lens, log_prob


class ReinforceTrainer(nn.Module):
    def __init__(self, args, speaker: nn.Module, listener: nn.Module):
        self.speaker = speaker
        self.listener = speaker
        self.i2w = args.i2w
        self.w2i = args.w2i
        self.D_hid = args.D_hid
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, image: torch.Tensor, distractor_images: torch.Tensor):
        batch_size = image.shape[0]

        msg, msg_lens, log_prob = self.speaker(image)
        msg = msg.detach()

        h_pred = self.listener.forward(msg, msg_lens.cpu())
        h_pred = h_pred.unsqueeze(1).repeat(1, 1 + len(distractor_images), 1)

        all_images = len(distractor_images) + 1
        img_idx = [list(range(all_images)) for _ in range(batch_size)]
        for c in img_idx:
            random.shuffle(c)

        target_idx = torch.tensor(
            np.argmax(np.array(img_idx) == 0, -1), dtype=torch.long,
            device=device)

        h_img = [self.beholder(image)] + [self.beholder(img)
                                          for img in distractor_images]
        h_img = torch.stack(h_img, dim=0).permute(1, 0, 2)
        for i in range(batch_size):
            h_img[i] = h_img[i, img_idx[i], :]

        logits = 1 / torch.mean(torch.pow(h_pred - h_img, 2),
                                2).view(-1, 1 + len(distractor_images))
        pred_outs = torch.argmax(logits, dim=-1).cpu().numpy()
        batch_inds = target_idx.cpu().numpy()

        acc = np.mean(np.equal(batch_inds, pred_outs))
        reward = np.equal(batch_inds, pred_outs) - acc
        loss = -torch.mean(log_prob * reward)
        return acc, loss
    
    def update(self, image, distractor_images):
        self.train()
        acc, loss = self.get_loss_acc(image, distractor_images)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return acc, loss


class SpeakerListener(nn.Module):

    def __init__(self, args):
        super(SpeakerListener, self).__init__()
        self.beholder = Beholder(args)
        self.speaker = Speaker(args, self.beholder)
        self.listener = Listener(args, self.beholder)
        self.i2w = args.i2w
        self.w2i = args.w2i
        self.D_hid = args.D_hid
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, image, spk_list=-1):
        msg, msg_lens = self.speaker.forward(image)

        if spk_list == 0:
            msg = msg.detach()

        msg_lens, sorted_indices = torch.sort(msg_lens, descending=True)
        msg = msg.index_select(0, sorted_indices)
        image = image.index_select(0, sorted_indices)

        h_pred = self.listener.forward(msg, msg_lens.cpu())
        return h_pred, image

    def get_loss_acc(self, image, distractor_images, spk_list=-1):
        batch_size = image.shape[0]

        h_pred, image = self.forward(image, spk_list)
        h_pred = h_pred.unsqueeze(1).repeat(1, 1 + len(distractor_images), 1)

        all_images = len(distractor_images) + 1
        img_idx = [list(range(all_images)) for _ in range(batch_size)]
        for c in img_idx:
            random.shuffle(c)

        target_idx = torch.tensor(
            np.argmax(np.array(img_idx) == 0, -1), dtype=torch.long, device=device)

        h_img = [self.beholder(image)] + [self.beholder(img)
                                          for img in distractor_images]
        h_img = torch.stack(h_img, dim=0).permute(1, 0, 2)
        for i in range(batch_size):
            h_img[i] = h_img[i, img_idx[i], :]

        logits = 1 / torch.mean(torch.pow(h_pred - h_img, 2),
                                2).view(-1, 1 + len(distractor_images))
        pred_outs = torch.argmax(logits, dim=-1).cpu().numpy()
        batch_inds = target_idx.cpu().numpy()

        acc = np.mean(np.equal(batch_inds, pred_outs))
        loss = self.loss_fn(logits, target_idx)
        return acc, loss

    def update(self, image, distractor_images):
        self.eval()
        acc, loss = self.get_loss_acc(image, distractor_images)
        return acc, loss

    def update(self, image, distractor_images, spk_list=-1):
        self.train()
        acc, loss = self.get_loss_acc(image, distractor_images, spk_list)
        self.optimizer.zero_grad()
        loss.backward()
        return acc, loss
