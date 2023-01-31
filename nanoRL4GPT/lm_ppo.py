import torch.nn

from conf import Config
from collector import LMCollector
from reward import BaseReward, Reward
from lm_policy import ACLM
from torch.utils.tensorboard import SummaryWriter


class PPO(torch.nn.Module):
    def __init__(self, lm_model: ACLM, reward: BaseReward, clip=0.1, logit_post_fn=None):
        super().__init__()
        self.lm_model = lm_model
        self.clip_range = clip
        self.reward_model = reward
        self.gen_steps = 0
        self.train_steps = 0
        self.dist = torch.distributions.Categorical

        if logit_post_fn:
            self.logit_post_fn = logit_post_fn
        else:
            self.logit_post_fn = torch.nn.Softmax(dim=-1)

    def forward(self, samples):
        # return loss
        # samples: step-wise tuples (states, action, reward, logit, ref_logit, value)
        states, actions, advantages, returns, gen_log_probs, eos_indices = samples
        logits = self.lm_model.forward_policy(states)
        logits = torch.squeeze(torch.take_along_dim(logits, eos_indices[:, None, None], dim=1), dim=1)
        cur_probs = self.logit_post_fn(logits)
        entropy = self.dist(probs=cur_probs).entropy()
        cur_probs = torch.squeeze(torch.gather(cur_probs, 1, actions[:, None]), dim=1)
        value = self.lm_model.forward_value(states, eos_indices)
        prob_ratio = torch.exp(torch.log(cur_probs) - gen_log_probs)
        policy_loss = -torch.mean(torch.min(advantages * prob_ratio,
                                            advantages * torch.clamp(prob_ratio, 1 - self.clip_range, 1 + self.clip_range)))

        value_loss = torch.mean(torch.nn.MSELoss()(value, returns))
        entropy_loss = - torch.mean(entropy)
        return policy_loss, value_loss, entropy_loss

    @torch.no_grad()
    def eval(self, input_ids, eos_token, reward_model: BaseReward):
        setences, _, terminal_pos = self.lm_model.generate(input_ids, logit_fn=self.logit_post_fn)
        rewards = reward_model.forward(setences, terminal_pos)
        return rewards

    @torch.no_grad()
    def generate_episode(self,
                         prompt_ids: torch.Tensor,
                         max_sentence_len: int,
                         top_k: int,
                         eos_token: int = 0,
                         ):
        batch = prompt_ids.shape[0]
        start_index = prompt_ids.shape[1]
        # max_step = max_sentence_len - start_index
        sentences, gen_probs, terminal_pos = self.lm_model.generate(prompt_ids, max_sentence_len, top_k, eos_token, self.logit_post_fn)

        values = self.lm_model.forward_value(sentences).view(*sentences.shape).detach().cpu()
        rewards = self.reward_model.calc_reward(sentences, torch.LongTensor(terminal_pos).to(sentences.device)).detach().cpu()

        ref_logits = self.lm_model.forward_ref(sentences)[:, start_index-1:-1, :]
        ref_logits = self.logit_post_fn(ref_logits, do_mask=False)
        ref_probs = torch.gather(ref_logits, 2, sentences[:, start_index:, None]).view(batch, -1)

        # detach
        input_ids = sentences.detach().cpu()
        gen_log_probs = torch.log(gen_probs).detach().cpu()
        ref_log_probs = torch.log(ref_probs).detach().cpu()

        return [(input_ids[i], start_index, terminal_pos[i],
                 gen_log_probs[i], ref_log_probs[i], values[i], rewards[i])
                for i in range(batch)]


def softmax_fn(temp=1.0, mask_ids=[]):
    def softmax_with_temper(logits: torch.FloatTensor, do_mask=True):
        if do_mask:
            for x in mask_ids: # mask unk and pad token
                logits[:, x].fill_(-1000.0)
        logits = logits/temp
        return torch.nn.Softmax(dim=-1)(logits)
    return softmax_with_temper


if __name__ == "__main__":
    pass