import os

RUNS_DIRECTORY = "runs_baselines/" 
PARTITION = "main"

def write_run_file(content, num): 
    os.makedirs(RUNS_DIRECTORY, exist_ok=True)    
    f = open(f"{RUNS_DIRECTORY}/run_{num}.job", "a")
    f.write(content)
    f.close()

# 1. SLURM header for the single script:
file = f"""#!/bin/bash
#SBATCH --account=prasanna_1363
#SBATCH --partition={PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00 

conda activate dcg
module load gcc/11.3.0 git/2.36.1

echo "Starting parallel job script"
"""

ENVS = ["pogema", "sc2_gen_protoss"] 
ALGOS = ["gacg", "dcg", "dicg", "vast", "qmix", "vdn"] 
MAX_SEEDS = 5 

"""
Baselines 
""" 
commands = []
count = 0 
for seed in range(MAX_SEEDS): 
    for env in ENVS: 
         for algo in ALGOS: 
            cmd = f"""python3 src/main.py --config={algo} --env-config={env} with seed={seed} use_cuda=False""" 
            count+=1
            write_run_file(file+cmd, count) 