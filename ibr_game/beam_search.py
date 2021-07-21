import torch
from torch import nn

import tqdm

from typing import Callable, Generator

HUGE = 1e15


def beam_search(model_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                beam_size: int,
                max_len: int,
                eos_id: int,
                bos_id: int,
                dataloader: Generator[torch.Tensor, None, None]):

    return_outputs = []
    return_logprobs = []

    for batch in dataloader:
        device = batch.get_device()
        batch_size = batch.size()[0]
        beam_outputs = torch.full(
            (batch_size, 1, 1), bos_id, dtype=torch.long).to(device)
        beam_inputs = torch.full(
            (batch_size, ), bos_id, dtype=torch.long).to(device)
        beam_hiddens = batch
        beam_logprobs = torch.zeros(batch_size, 1).to(device)
        finish_mask = torch.zeros(batch_size, 1).to(device)

        for i in range(max_len):
            outputs, beam_hiddens_ = model_func(beam_hiddens, beam_inputs)
            vocabulary = outputs.size()[-1]

            # (B, b) -> (B, b, V)
            beam_logprobs = beam_logprobs.unsqueeze(
                2).repeat(1, 1, vocabulary)
            # (B x b, V) -> (B, b, V)
            outputs = outputs.view(beam_logprobs.size())

            finish_mask = finish_mask.unsqueeze(2).repeat(1, 1, vocabulary)
            outputs = outputs * (1 - finish_mask) - HUGE * finish_mask
            outputs[:, :, eos_id] = outputs[:, :, eos_id] * \
                (1 - finish_mask[:, :, 0])

            beam_logprobs = (beam_logprobs + outputs).view(batch_size, -1)
            beam_logprobs, indices = torch.topk(beam_logprobs, beam_size)

            beam_indices = indices // vocabulary
            word_indices = indices % vocabulary
            beam_inputs = word_indices.view(-1)
            finish_mask = (word_indices == eos_id).float()

            # (B, b, i+1) -> (B, b, i+1)
            beam_outputs = torch.gather(
                beam_outputs, 1, beam_indices.unsqueeze(2).repeat(1, 1, i+1))
            # cat((B, b, i+1), (B, b, 1)) -> (B, b, i+2)
            beam_outputs = torch.cat(
                [beam_outputs, word_indices.unsqueeze(2)], dim=2)
            # (B, b, H) -> (B, b, H) -> (B x b, H)
            hid_size = beam_hiddens_.size()[-1]
            beam_hiddens = torch.gather(
                beam_hiddens_.view(batch_size, -1, hid_size),
                1,
                beam_indices.unsqueeze(2).repeat(1, 1, hid_size))\
                .view(-1, hid_size)

        return_outputs.append(beam_outputs)
        return_logprobs.append(beam_logprobs)

    return_outputs = torch.cat(return_outputs, dim=0)
    return_logprobs = torch.cat(return_logprobs, dim=0)

    return (return_outputs, return_logprobs)


class LanguageModel(nn.Module):
    def __init__(self, vocabulary, hidden_size):
        super(LanguageModel, self).__init__()
        self.word_emb = nn.Embedding(vocabulary, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.output = nn.Sequential(nn.Linear(hidden_size, vocabulary),
                                    nn.LogSoftmax(dim=-1))

    def forward(self, hidden, inputs):
        hid = self.gru(self.word_emb(inputs), hidden)
        return self.output(hid), hid


def generating_random_data(batch_size: int, hidden_size: int, size: int):
    for i in tqdm.tqdm(range(size)):
        yield torch.randn(batch_size, hidden_size).cuda()


if __name__ == "__main__":
    language_model = LanguageModel(200, 128).cuda()
    with torch.no_grad():
        beam_search(language_model, 10, 15, 99, 0,
                    generating_random_data(128, 128, 1000))
