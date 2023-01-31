from transformers import GPT2PreTrainedModel
import torch


class BaseReward(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def calc_reward(self, input_ids, *args):
        raise Exception("not impl")


class Reward(BaseReward):
    def __init__(self, model: GPT2PreTrainedModel):
        super().__init__()
        self.model = model
        self.header = torch.nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, eos_indices=None, attention_mask=None):
        hidden_state = self.model(input_ids, attention_mask).last_hidden_state
        eos_indices = eos_indices[:, None, None]
        hidden_state = torch.take_along_dim(hidden_state, eos_indices, dim=1)
        hidden_state = torch.squeeze(hidden_state, dim=1)
        scores = torch.nn.Sigmoid()(self.header(hidden_state))
        return torch.squeeze(scores, dim=-1)

    @torch.no_grad()
    def calc_reward(self, input_ids, eos_indices=None):
        return self.forward(input_ids, eos_indices)


class CounterReward(BaseReward):
    def __init__(self, shortest=True):
        super().__init__()
        self.shortest = shortest

    @torch.no_grad()
    def calc_reward(self, input_ids, eos_indices=None):
        batch = input_ids.shape[0]
        size = input_ids.shape[1]
        if self.shortest:
            return 512 - torch.clone(eos_indices.float())
        else:
            return torch.clone(eos_indices.float())


def train_reward():
    pass


def gen_samples():
    pass


if __name__ == "__main__":
    # test reward of GPT2
    pass