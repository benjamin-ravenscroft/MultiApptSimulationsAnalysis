import numpy as np
import pandas as pd
import subprocess
import os
from itertools import permutations, combinations, combinations_with_replacement

# define summarization functions
def get_summary(df):
    res = pd.DataFrame(columns=['class', 'param', 'value'])
    res.loc[res.shape[0]] = [-1, 'age_out', df['age_out'].mean()]
    for cls in sorted(df['class'].unique()):
        res.loc[res.shape[0]] = [cls, 'age_out', df.loc[df['class'] == cls, 'age_out'].mean()]
    res.loc[res.shape[0]] = [-1, 'wait_time', df.loc[df['n_appts'] > 0, 'total_wait_time'].mean()]
    for cls in sorted(df['class'].unique()):
        res.loc[res.shape[0]] = [cls, 'wait_time', df.loc[(df['class'] == cls) & (df['n_appts'] > 0), 'total_wait_time'].mean()]
    return res

def get_mean_summary(df, arr_lam, arrival_probabilities, pathways, wait_effects, modality_effects, modality_policies):
    res = pd.DataFrame(columns=['class', 'param', 'mean', 'sem'])
    for cls in sorted(df['class'].unique()):
        res.loc[res.shape[0]] = [cls, 'age_out', df.loc[(df['class']==cls) & (df['param']=='age_out'), 'value'].mean(),
                                 df.loc[(df['class']==cls) & (df['param']=='age_out'), 'value'].std()/np.sqrt(df.loc[(df['class']==cls) & (df['param']=='age_out'), 'value'].shape[0])]
        res.loc[res.shape[0]] = [cls, 'wait_time', df.loc[(df['class']==cls) & (df['param']=='wait_time'), 'value'].mean(),
                                 df.loc[(df['class']==cls) & (df['param']=='wait_time'), 'value'].std()/np.sqrt(df.loc[(df['class']==cls) & (df['param']=='wait_time'), 'value'].shape[0])]

    # add parameters
    for cls in range(len(pathways)):
        res[f"arr_lam_{cls}"] = arr_lam*arrival_probabilities[cls]
        res[f"base_duration_{cls}"] = pathways[cls]
        res[f"wait_effect_{cls}"] = wait_effects[cls]
        res[f"modality_effect_{cls}"] = modality_effects[cls]
        res[f"modality_policy_{cls}"] = modality_policies[cls]
    return res

# define path to C executable
program_path = "/home/benja/kidsAbility/MultiApptSimulations/build/simulation"

# set program parameters
runs = 5
epochs = 10000
clinicians = 120
max_caseload = 4
arr_lam = 10
pathways = [7,10,13]
wait_effects = [1.2, 1.2, 1.2]
modality_effects = [0.5, 0.0, -0.5]
moadlity_policies = [0.5, 0, 1]
max_ax_age = 3
age_params = [1.5, 1]
priority_order = [0,1,2]
priority_wlist = "true"
arrival_probabilities = [0.3333, 0.3333, 0.3333]

folder = "test/"

# define the program call
program_call = program_path
program_call += f" --runs={runs} --n_epochs={epochs} --clinicians={clinicians} --max_caseload={max_caseload} --arr_lam={arr_lam} --folder={folder}"
program_call += f" --pathways={','.join(map(str, pathways))} --wait_effects={','.join(map(str, wait_effects))} --modality_effects={','.join(map(str, modality_effects))} --modality_policies={','.join(map(str, moadlity_policies))}"
program_call += f" --max_ax_age={max_ax_age} --age_params={','.join(map(str, age_params))} --priority_order={','.join(map(str, priority_order))} --priority_wlist={priority_wlist} --arrival_probs={','.join(map(str, arrival_probabilities))}"
print(program_call)

# call the program as a subprocess and wait
subprocess.call(program_call, shell=True)

# create summary file
output_folder = "/mnt/d/OneDrive - University of Waterloo/KidsAbility Research/Service Duration Analysis/C++ Simulations/test/"

temp = pd.DataFrame()
for i in range(runs):
    df = pd.read_parquet(output_folder + f"simulation_data_{i}.parquet")
    temp = pd.concat([temp, get_summary(df)], axis=0)
    os.remove(output_folder + f"simulation_data_{i}.parquet")   # remove the file after reading

summary = get_mean_summary(temp, arr_lam, arrival_probabilities, pathways, wait_effects, modality_effects, moadlity_policies)
del temp
print(summary)
summary.to_parquet(output_folder + "summary.parquet")

# check the dimension of the parameter search space
wait_effects = [-1.2, 0, 1.2]
modality_effects = [-1, 0, 1]
modality_policies = [0, 0.25, 0.5, 0.75, 1]

count = 0
for combination in combinations_with_replacement(wait_effects, 3):
    for combination2 in combinations_with_replacement(modality_effects, 3):
        for combination3 in combinations_with_replacement(modality_policies, 3):
            count += 1
print(f"Total number of combinations: {count}")
est_runtime = 2/60 * runs
print(f"Estimated runtime (concurrent): {count*est_runtime} minutes.")
parallelization = 30
print(f"Estimated runtime (parallel): {count*est_runtime/parallelization} minutes.")
