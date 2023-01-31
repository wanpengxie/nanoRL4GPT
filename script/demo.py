import torch.optim
from transformers import GPT2LMHeadModel, GPT2Model, BertTokenizer
import time

from nanoRL4GPT.lm_ppo import PPO, softmax_fn
from nanoRL4GPT.lm_policy import ACLM
from nanoRL4GPT.reward import BaseReward, CounterReward
from nanoRL4GPT.collector import LMCollector
import tqdm
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 16
    epoch = 10
    inner_epoch = 1
    kl_coe = 0.1
    gamma = 0.99
    gae_lambda = 0.95
    clip = 0.2
    top_k = 200
    max_sentence_len = 128
    max_prompt_len = 16
    train_batch = 128
    lr = 5e-6
    buffer = 256
    device = "cuda"
    value_coe = 0.1
    entropy_coe = 0.005
    max_grad_norm = 0.5
    eos_token_id = 100

    writer = SummaryWriter("logger_{0}".format(int(time.time())))

    collector = LMCollector(buffer, kl_coe, gamma, gae_lambda)
    reward = CounterReward()

    policy = GPT2LMHeadModel.from_pretrained('gpt2')
    value = GPT2Model.from_pretrained('gpt2')
    ref = GPT2LMHeadModel.from_pretrained('gpt2')
    ac_policy = ACLM(policy, value, ref).to(device)

    ppo = PPO(ac_policy, reward, clip, logit_post_fn=softmax_fn(mask_ids=[0, 100]))
    params = list(ac_policy.policy_model.parameters()) + list(ac_policy.value_model.parameters())
    opt = torch.optim.AdamW(params, lr=lr)

    dataset = [torch.randint(110, 1000, [10, 5]) for i in range(10)]

    sample_step = 0
    train_step = 0
    for i in range(epoch):
        # tdata = tqdm.tqdm(dataset(device=device), total=datasize)
        for prompts in dataset:
            sample_step += 1
            if collector.is_full():
                collector.calc_gae()
                print ("episodes summary info: ", collector.summary())
                ac_policy.train()
                for samples in collector.sample(inner_epoch, batch=train_batch, device=device):
                    train_step += 1
                    ploss, vloss, eloss = ppo.forward(samples)
                    loss = ploss + vloss * value_coe + entropy_coe * eloss
                    loss = loss.mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                    opt.step()
                    opt.zero_grad()
                    ploss_eval = ploss.mean().detach().item()
                    vloss_eval = vloss.mean().detach().item()
                    eloss_eval = eloss.mean().detach().item()
                    total_loss = loss.mean().detach().item()
                    writer.add_scalars("loss", {"total_loss": total_loss,
                                                "policy_loss": ploss_eval,
                                                "value_loss": vloss_eval,
                                                "entropy_loss": eloss_eval,
                                                },
                                       global_step=train_step)
                collector.reset()
                ac_policy.eval()

            ac_policy.eval()
            episodes = ppo.module.generate_episode(prompts, max_sentence_len, top_k, eos_token_id)
            cur_rewards = torch.mean(torch.stack([x[-1] for x in episodes])).detach().item()
            writer.add_scalar("eval_reward", cur_rewards, global_step=sample_step)
            collector.add_buffer(episodes)