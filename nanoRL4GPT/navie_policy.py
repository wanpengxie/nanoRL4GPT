import torch
from transformers import GPT2LMHeadModel
from reward import BaseReward


class RandomSelectPolicy(torch.nn.Module):
    def __init__(self, sft: GPT2LMHeadModel, reward: BaseReward, best_select=-1):
        super(RandomSelectPolicy, self).__init__()
        self.policy_model = sft
        self.reward = reward
        self.selector = best_select

    def forward_policy(self, input_ids):
        logits = self.policy_model.forward(input_ids).logits
        return logits

    def forward_value(self, input_ids, eos_indice=None):
        if eos_indice is None:
            fake_value = torch.randn(*input_ids.shape)
        else:
            fake_value = torch.randn(input_ids.shape[0])
        return fake_value

    def forward_ref(self, input_ids):
        return self.forward_policy(input_ids)

    @torch.no_grad()
    def generate(self, input_ids, max_len=128, top_k=20, eos_token=0, logit_fn=torch.nn.Softmax(dim=-1)):
        if self.selector < 0:
            return self.generate_random(input_ids, max_len, top_k, eos_token, logit_fn)
        else:
            batch = input_ids.shape[0]
            res = []
            best_s = []
            best_t = []
            for i in range(self.selector):
                sentences, probs, eos_pos = self.generate_random(input_ids, max_len, top_k, eos_token, logit_fn)
                res.append((sentences, probs, eos_pos))
            for i in range(batch):
                si = torch.cat([row[0][i:i+1] for row in res], dim=0).to(input_ids.device)
                ti = torch.LongTensor([row[2][i] for row in res]).to(input_ids.device)
                reward = self.reward.calc_reward(si, ti)
                print (si)
                print (reward)
                ax = torch.argmax(reward)
                sx = si[ax][None, :]
                tx = ti[ax]
                print (sx)
                print (ax)
                best_s.append(sx)
                best_t.append(tx)
            return torch.cat(best_s, dim=0).to(input_ids.device), probs, best_t


    def generate_random(self, input_ids, max_len=128, top_k=20, eos_token=0, logit_fn=torch.nn.Softmax(dim=-1)):
        batch = input_ids.shape[0]
        start_index = input_ids.shape[1]
        gen_probs = torch.zeros((batch, max_len-start_index))
        terminal_pos = [max_len-1 for x in range(batch)]
        terminal_flags = [False for x in range(batch)]

        for i in range(start_index, max_len):
            logits = self.forward_policy(input_ids)[:, -1, :]
            logits = logit_fn(logits)

            # sampler
            top_score, top_index = torch.topk(logits, top_k, dim=1, largest=True, sorted=True)
            next_tokens_index = torch.multinomial(top_score, num_samples=1)
            next_tokens = torch.gather(top_index, 1, next_tokens_index)
            next_tokens_prob = torch.gather(top_score, 1, next_tokens_index)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            next_token_ids = next_tokens.detach().cpu()

            for k in range(batch):
                if terminal_pos[k] == max_len-1 and next_token_ids[k][0].item() == eos_token:
                    terminal_pos[k] = i
                    terminal_flags[k] = True
            gen_probs[:, i-start_index] = torch.clone(next_tokens_prob.view(-1))
            if all(terminal_flags):
                break
        return input_ids, gen_probs, terminal_pos


class GroudTruthPolicy(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def forward_policy(self, input_ids):
        shape = list(input_ids.shape)
        shape.append(30000)
        return torch.randn(*shape)

    def forward_value(self, input_ids, eos_indice=None):
        if eos_indice is None:
            fake_value = torch.randn(*input_ids.shape)
        else:
            fake_value = torch.randn(input_ids.shape[0])
        return fake_value

    def forward_ref(self, input_ids):
        return self.forward_policy(input_ids)

    def generate(self, input_ids, max_len=128, top_k=20, eos_token=0, logit_fn=torch.nn.Softmax(dim=-1)):
        sentences, terminal_pos = self.dataset.get(input_ids, max_len)
        return sentences, torch.randn(sentences.shape[0], sentences.shape[1]), terminal_pos


if __name__ == "__main__":
    from transformers import PreTrainedModel, GPT2Config, GPT2LMHeadModel, GPT2Model
    from reward import Reward

    # test policy
    torch.manual_seed(42)
    config = GPT2Config()
    policy = GPT2LMHeadModel.from_pretrained("gpt2")

    model = GPT2Model(config=config)
    reward_model = Reward(model).eval()

    lm_policy = RandomSelectPolicy(policy, reward_model, 10).eval()
    input_ids = torch.randint(2, 10, (3, 2))
    print (input_ids)
    s, p, t = lm_policy.generate(input_ids, max_len=10)
    print (s)
    print (p)
    print (t)
    r = reward_model.calc_reward(s, torch.LongTensor(t))
    print (r)


