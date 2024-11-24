import subprocess
import os
from itertools import combinations_with_replacement, permutations, product
import time
from dotenv import dotenv_values

# load folder locations
config = dotenv_values("./.env")

# define simulation parameterization
runs = 5
epochs = 10000
clinicians = 80
max_caseload = 4
arr_lam = 10
pathways = (7,10,13)
wait_effects = [0, 0, 0]
modality_effects = [-0.5, 0, 0.5]
modality_policies = [0.0, 0.25, 0.5, 0.75, 1]
max_ax_age = 3
age_params = (1.5, 1)
priority_order = (0,1,2)
priority_wlist = "false"
arrival_probabilities = (0.3333, 0.3333, 0.3333)

# base_folder = "/mnt/d/OneDrive - University of Waterloo/KidsAbility Research/Service Duration Analysis/C++ Simulations/"
base_folder = config["BATCH_OUTPUT_BASE_DIR"]
output_folder = base_folder + "batch_summary_fcfs_perm_05.parquet/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

processes = []
i = 0
start = time.time()
# for combination in permutations(wait_effects, 3):
for combination in [wait_effects]:
    # for combination2 in [modality_effects]:
    #     for combination3 in [modality_policies]:
    for combination2 in permutations(modality_effects, 3):
        for combination3 in product(modality_policies, modality_policies, modality_policies):
            sim_name = f"sim_{i}"
            folder = base_folder + "batch_sims_fcfs_perm_05/" + sim_name + "/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            call = ["python"]
            call += [config["CLI_APP_LOCATION"]]
            call += ["--program-path", config["CPP_LOCATION"]]
            call += ["--sim-name", f"{sim_name}"]
            call += ["--directory", "batch_sims"]
            call += ["--runs", f"{runs}"]
            call += ["--epochs", f"{epochs}"]
            call += ["--clinicians", f"{clinicians}"]
            call += ["--arr-lam", f"{arr_lam}"]
            call += ["--pathways"] + list(map(str, pathways))
            call += ["--wait-effects"] + list(map(str, combination))
            call += ["--modality-effects"] + list(map(str, combination2))
            call += ["--modality-policies"] + list(map(str, combination3))
            call += ["--priority-wlist", f"{priority_wlist}"]
            call += ["--folder", folder]
            call += ["--output-folder", output_folder]
            
            processes.append(subprocess.Popen(call))
            i += 1
        exit_codes = [p.wait() for p in processes]
        processes = []
    
if len(processes) > 0:
    exit_codes = [p.wait() for p in processes]
    processes = []
end = time.time()
print(f"Finished simulations with exit codes: {exit_codes}")
print(f"Time taken: {(end - start)/60}min {(end - start)%60}s.")
