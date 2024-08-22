import os 

def baselines(): 
    CONFIGS = ["dicg", "dcg"]
    ENVS = ["gather", "hallway", "pursuit", "disperse"] 
    SEEDS = 3 
    PARALLEL = False 

    for s in range(SEEDS): 
        for e in ENVS:
            for a in CONFIGS: 
                command = f"python src/main.py --config={a} --env-config={e} with use_cuda=False seed={s}" 
                if PARALLEL: command += " &" 
                os.system(command) 
baselines() 