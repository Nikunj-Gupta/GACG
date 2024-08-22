all: 
	clear 
	CUDA_VISIBLE_DEVICES=3 python src/main.py --config=dicg --env-config=gather with seed=0 use_cuda=False 
