# LM models of actor critor model
import torch.nn
from transformers import GPT2PreTrainedModel
from conf import Config
from torch import nn


class ACLM(nn.Module):
    def __init__(self, policy: GPT2PreTrainedModel, value: GPT2PreTrainedModel, ref: GPT2PreTrainedModel):
        super().__init__()
        self.policy_model = policy
        self.value_model = value
        self.ref_model = ref
        self.ref_model.eval()
        self.setup_value_head()

    def train(self, mode=True):
        # always keep ref_model eval
        super().train(mode)
        self.ref_model.eval()

    def setup_value_head(self):
        self.value_header = torch.nn.Linear(self.value_model.config.hidden_size, 1, bias=False)

    def generate(self, input_ids, max_len=128, top_k=200, eos_token=0, logit_fn=torch.nn.Softmax(dim=-1)):
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
            # if i-start_index < 10: # debug info
            #     print ('-----------')
            #     print (top_score[:50])
            #     print (top_index[:50])
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

    def forward_policy(self, input_ids):
        logits = self.policy_model.forward(input_ids).logits
        return logits

    def get_input_logits(self, input_ids, logits):
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1: , None]
        logits = torch.gather(logits, 2, targets).view(input_ids.shape[0], input_ids.shape[1]-1)
        return logits

    def forward_ref(self, input_ids):
        logits = self.ref_model.forward(input_ids).logits
        return logits

    def forward_value(self, input_ids, eos_indices=None):
        batch = input_ids.shape[0]
        values = self.value_model.forward(input_ids).last_hidden_state
        if isinstance(eos_indices, torch.Tensor):
            values = torch.take_along_dim(values, eos_indices[:, None, None], dim=1)
            values = values.view(batch, self.value_model.config.hidden_size)
        elif eos_indices == -1:
            values = values[:, -1, :]
        else:
            pass
        values = self.value_header(values)
        return torch.squeeze(values, dim=-1)