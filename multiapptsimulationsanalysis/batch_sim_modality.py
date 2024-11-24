import subprocess
import os
from itertools import combinations_with_replacement, permutations, product
import time
from dotenv import dotenv_values
import numpy as np

# load folder locations
config = dotenv_values("./.env")

# define simulation parameterization
runs = 5
epochs = 10000
clinicians = 80
max_caseload = 4
arr_lam = 10
pathways = [15]
wait_effects = [0]
modality_effects = np.round(np.arange(0, -0.5-0.05, -0.05),2)
modality_policies = np.round(np.arange(0, 1+0.1, 0.1),2)
max_ax_age = 3
age_params = (1.5, 1)
priority_order = [0]
priority_wlist = "false"
arrival_probabilities = [1]
virtual_att_probs = [0.9,0.025,0.025,0.05]
face_att_probs = [0.8,0.05,0.05,0.1]
att_gap = list(np.arange(0, -0.5-0.05, -0.05))

# base_folder = "/mnt/d/OneDrive - University of Waterloo/KidsAbility Research/Service Duration Analysis/C++ Simulations/"
base_folder = config["BATCH_OUTPUT_BASE_DIR"]
output_folder = base_folder + "batch_summary_mod.parquet/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

processes = []
i = 0
start = time.time()

for gap in att_gap:
    for eff in modality_effects:
        for pol in modality_policies:
            # set simulation name
            sim_name = f"sim_{i}"
            folder = base_folder + "batch_sims_mod/" + sim_name + "/"
            if not os.path.exists(folder):
                os.makedirs(folder)

            # create the face_att_probs for simulation based on gap
            face_att_probs = [
                virtual_att_probs[0] + gap,
                (1 - (virtual_att_probs[0]+gap))/4,
                (1 - (virtual_att_probs[0]+gap))/4,
                (1 - (virtual_att_probs[0]+gap))/2
            ]
            face_att_probs = [np.round(x, 4) for x in face_att_probs]   # round the list

            # calculate the modality effect in terms of number of appointments rather than percent
            mod_eff = eff*pathways[0]

            # set the simulation parameters
            call = ["python"]
            call += [config["CLI_APP_LOCATION"]]
            call += ["--program-path", config["CPP_LOCATION"]]
            call += ["--sim-name", f"{sim_name}"]
            call += ["--directory", "batch_sims"]
            call += ["--runs", f"{runs}"]
            call += ["--epochs", f"{epochs}"]
            call += ["--clinicians", f"{clinicians}"]
            call += ["--arr-lam", f"{arr_lam}"]
            for t in pathways:
                call += ["--pathways", str(t)]
            for t in wait_effects:
                call += ["--wait-effects", str(t)]
            call += ["--modality-effects", str(mod_eff)]
            call += ["--modality-policies", str(pol)]
            call += ["--priority-order", str(0)]
            call += ["--priority-wlist", f"{priority_wlist}"]
            for t in arrival_probabilities:
                call += ["--arrival-probabilities", str(t)]
            call += ["--folder", folder]
            call += ["--output-folder", output_folder]
            call += ["--virtual-att-probs"] + list(map(str, virtual_att_probs))
            call += ["--face-att-probs"] + list(map(str, face_att_probs))

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
