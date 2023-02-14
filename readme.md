一个玩具版本的gpt2 rlhf项目，传说中chatgpt的优化方式，核心代码大概500行

动机来自于 https://github.com/karpathy/nanoGPT，主要为了验证一些instructGPT论文中的核心特性（reward函数的泛化性，ppo的收敛性，generate policy和pretrain policy的权衡）

主要依赖

>pytorch>=1.14.0\
>transformers >= 4.18.0

实现参考论文 https://arxiv.org/abs/2203.02155

RL4LM更完整的框架参考：https://github.com/allenai/RL4LMs

一个很棒的RL框架：https://github.com/thu-ml/tianshou

demo 见 script/demo.py
