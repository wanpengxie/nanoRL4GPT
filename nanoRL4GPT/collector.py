# env with reward function
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class LMCollector(object):
    def __init__(self, max_epsiodes, kl_coe, gamma=0.9, gae_lambda=0.9, pad_id=100):
        self.max_epsiodes = max_epsiodes
        self.kl_coe = kl_coe
        self.pad_id = pad_id
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def is_full(self) -> bool:
        return self.current_num >= self.max_epsiodes

    def reset(self):
        self.current_num = 0
        self.episodes = []
        self.samples = []

    def add_buffer(self, episodes):
        self.episodes += episodes
        self.current_num += len(episodes)

    def sample(self, epoch: int, batch=2, shuffle=True, device="cpu"):
        for i in range(epoch):
            if shuffle:
                np.random.shuffle(self.samples)
            for j in range(len(self.samples)//batch):
                samples = self.samples[j*batch:(j+1)*batch]
                yield self.pad_samples(samples, device)

    def pad_samples(self, samples, device):
        sentences = pad_sequence([x[0] for x in samples], batch_first=True, padding_value=self.pad_id)
        actions = torch.LongTensor([x[1] for x in samples])
        advantages = torch.FloatTensor([x[2] for x in samples])
        returns = torch.FloatTensor([x[3] for x in samples])
        gen_log_probs = torch.FloatTensor([x[4] for x in samples])
        indices = torch.LongTensor([len(x[0])-1 for x in samples])
        return (sentences.to(device), actions.to(device),
                advantages.to(device), returns.to(device),
                gen_log_probs.to(device), indices.to(device))

    def summary(self):
        ep_size = len(self.episodes)
        sample_size = len(self.samples)
        terminals = torch.mean(torch.Tensor([x[2] for x in self.episodes]))
        rewards = sum([x[-1] for x in self.episodes])/ep_size
        return {"ep_size": ep_size, "sample_size": sample_size, "ep_len": terminals, "rewards": rewards}

    def calc_gae(self):
        for episode in self.episodes:
            # sentence = [id0, id1, id2, ...., idT=eos, pad, pad, ...]
            # T = terminal_index,
            # max_step = len(gen_probs) = len(ref_probs) = len(sentence) - start_index
            # convert to samples: states, action, advantage, return, gen_log_prob
            sentence, start_index, terminal_index, gen_log_probs, ref_log_probs, values, reward = episode
            advantage = 0.0
            for i in range(terminal_index, start_index-1, -1):
                states = sentence[:i]
                action = sentence[i]
                cur_value, next_value = values[i-1], 0 if i == terminal_index else values[i]
                gen_log_prob, ref_log_prob = gen_log_probs[i-start_index], ref_log_probs[i-start_index]
                kl_r = - self.kl_coe * (gen_log_prob - ref_log_prob)
                reward_sum = kl_r + (i == terminal_index) * reward
                delta = reward_sum + next_value * self.gamma - cur_value
                advantage = delta + self.gamma * self.gae_lambda * advantage
                r = advantage + cur_value
                sample = (states, action, advantage, r, gen_log_prob)
                self.samples.append(sample)


if __name__ == "__main__":
    pass