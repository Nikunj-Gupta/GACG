## Run an experiment 

Tasks can be found in `src/envs`. 

To run experiments on SMAC benchmark:
```shell
python src/main.py --config=gacg --env-config=sc2 with env_args.map_name='10m_vs_11m' 
```

The requirements.txt file can be used to install the necessary packages into a virtual environment.

## Baselines used in this paper
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**DCG**: Deep Coordination Graphs](https://arxiv.org/abs/1910.00091)
- [**DICG**: Deep Implicit Coordination Graphs for Multi-agent Reinforcement Learning](https://arxiv.org/abs/2006.11438) 
- [**CASEC**: Context-Aware Sparse Deep Coordination Graphs](https://arxiv.org/abs/2106.02886)
- [**VAST**: VAST: Value Function Factorization with Variable Agent Sub-Teams](https://proceedings.neurips.cc/paper_files/paper/2021/hash/c97e7a5153badb6576d8939469f58336-Abstract.html)
- [**QTRAN**: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
