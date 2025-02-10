all: 
	clear 
	# python src/main.py --config=qmix --env-config=pogema with seed=10 use_cuda=True agent=rnn
	# python src/main.py --config=qmix --env-config=pogema with seed=10 use_cuda=True agent=gnn 
	# python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" seed=10 use_cuda=False 
	# python src/main.py --config=qtran --env-config=gymma with env_args.N=4 env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" seed=10 use_cuda=False 
	# python src/main.py --config=qtran --env-config=pogema with env_args.num_agents=8 seed=20 use_cuda=False 
	python src/main.py --config=qtran --env-config=gymma with env_args.N=3 env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" seed=10 use_cuda=False

pogema-runs: 
	# CUDA_VISIBLE_DEVICES=0 python src/main.py --config=dicg --env-config=pogema with seed=0 use_cuda=True & 
	# CUDA_VISIBLE_DEVICES=1 python src/main.py --config=vast --env-config=pogema with seed=0 use_cuda=True & 
	# CUDA_VISIBLE_DEVICES=2 python src/main.py --config=gacg --env-config=pogema with seed=0 use_cuda=True & 
	# CUDA_VISIBLE_DEVICES=3 python src/main.py --config=dcg --env-config=pogema with seed=0 use_cuda=True & 
	# CUDA_VISIBLE_DEVICES=0 python src/main.py --config=qmix --env-config=pogema with seed=0 use_cuda=True & 
	# CUDA_VISIBLE_DEVICES=0 python src/main.py --config=vdn --env-config=pogema with seed=0 use_cuda=True & 

smacv2-runs: 
	# CUDA_VISIBLE_DEVICES=0 python src/main.py --config=dicg --env-config=sc2_gen_protoss with seed=0 use_cuda=True 
	# CUDA_VISIBLE_DEVICES=0 python src/main.py --config=dcg --env-config=sc2_gen_protoss with seed=0 use_cuda=True 
	# CUDA_VISIBLE_DEVICES=0 python src/main.py --config=gacg --env-config=sc2_gen_protoss with seed=0 use_cuda=True 
	
run_baselines: 
	for f in runs_baselines/*.job; do sbatch $$f; done 
