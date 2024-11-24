import numpy as np
import pandas as pd
import subprocess
import os
from itertools import permutations, combinations, combinations_with_replacement
import typer
from typing_extensions import Annotated
from typing import Tuple, List

# define summarization functions
def get_summary(df):
    res = pd.DataFrame(columns=['class', 'param', 'value'])
    res.loc[res.shape[0]] = [-1, 'age_out', df['age_out'].mean()]
    for cls in sorted(df['class'].unique()):
        res.loc[res.shape[0]] = [cls, 'age_out', df.loc[df['class'] == cls, 'age_out'].mean()]
    res.loc[res.shape[0]] = [-1, 'wait_time', df.loc[df['n_appts'] > 0, 'total_wait_time'].mean()]
    for cls in sorted(df['class'].unique()):
        res.loc[res.shape[0]] = [cls, 'wait_time', df.loc[(df['class'] == cls) & (df['n_appts'] > 0), 'total_wait_time'].mean()]
    res.loc[res.shape[0]] = [-1, 'pct_face', df.loc[df['n_appts'] > 0, 'pct_face'].mean()]
    for cls in sorted(df['class'].unique()):
        res.loc[res.shape[0]] = [cls, 'pct_face', df.loc[(df['class'] == cls) & (df['n_appts'] > 0), 'pct_face'].mean()]
    return res

def get_mean_summary(df, runs, arr_lam, arrival_probabilities, pathways, wait_effects, modality_effects, modality_policies, virtual_att_probs, face_att_probs):
    res = pd.DataFrame(columns=['class', 'param', 'mean', 'sem'])
    for cls in sorted(df['class'].unique()):
        res.loc[res.shape[0]] = [cls, 'age_out', df.loc[(df['class']==cls) & (df['param']=='age_out'), 'value'].mean(),
                                 df.loc[(df['class']==cls) & (df['param']=='age_out'), 'value'].std()/np.sqrt(df.loc[(df['class']==cls) & (df['param']=='age_out'), 'value'].shape[0])]
        res.loc[res.shape[0]] = [cls, 'wait_time', df.loc[(df['class']==cls) & (df['param']=='wait_time'), 'value'].mean(),
                                 df.loc[(df['class']==cls) & (df['param']=='wait_time'), 'value'].std()/np.sqrt(df.loc[(df['class']==cls) & (df['param']=='wait_time'), 'value'].shape[0])]
        res.loc[res.shape[0]] = [cls, 'pct_face', df.loc[(df['class']==cls) & (df['param']=='pct_face'), 'value'].mean(),
                                 df.loc[(df['class']==cls) & (df['param']=='pct_face'), 'value'].std()/np.sqrt(df.loc[(df['class']==cls) & (df['param']=='pct_face'), 'value'].shape[0])]

    # add parameters
    for cls in range(len(pathways)):
        res['runs'] = runs
        res[f"arr_lam_{cls}"] = arr_lam*arrival_probabilities[cls]
        res[f"base_duration_{cls}"] = pathways[cls]
        res[f"wait_effect_{cls}"] = wait_effects[cls]
        res[f"modality_effect_{cls}"] = modality_effects[cls]
        res[f"modality_policy_{cls}"] = modality_policies[cls]
    # add attendance probability parameters
    for i, lab in enumerate(['attend', 'ns', 'lm_canc', 'adv_canc']):
        res[f"{lab}_virtual_p"] = virtual_att_probs[i]
        res[f"{lab}_face_p"] = face_att_probs[i]
    return res


def main(program_path: Annotated[str, typer.Option(help="Path to simulation executable")]="",
         sim_name: Annotated[str, typer.Option(help="Name of the simulation.")]="test",
         directory: Annotated[str, typer.Option(help="Directory to save simulation results.")]="",
         runs: Annotated[int, typer.Option(help="Number of runs.")]=5,
         epochs: Annotated[int, typer.Option(help="Number of epochs per run.")]=10000,
         clinicians: Annotated[int, typer.Option(help="Number of clincians.")]=120,
         max_caseload: Annotated[int, typer.Option(help="Max caseload per clinician. 1=weekly visits, 4=monthly, etc.")]=4,
         arr_lam: Annotated[int, typer.Option(help="Poisson process arrival rate.")]=10,
         pathways: Annotated[List[int], typer.Option(help="Base duration for pathways.")]=[7,10,13],
         wait_effects: Annotated[
             List[float],
             typer.Option(help="Wait effect per pathway (in terms of years).")
        ]=[1.2,1.2,1.2],
         modality_effects: Annotated[
             List[float], 
             typer.Option(help="Modality effect per pathway")
        ]=[0.5,0.0,-0.5],
         modality_policies: Annotated[
             List[float], 
             typer.Option(help="Modality policy for each pathway.")
        ]=[0.5,0,1],
         max_ax_age: Annotated[int, typer.Option(help="Max assessment age allowable.")]=3,
         age_params: Annotated[
             Tuple[float, float], 
             typer.Option(help="Arrival age parameters (mu, std for normal distribution).")
        ]=(1.5,1),
         priority_order: Annotated[
             List[int], 
             typer.Option(help="Priority order if using a priority-based waitlist.")
        ]=[0,1,2],
         priority_wlist: Annotated[
             str, 
             typer.Option(help="Flag to indicate if waitlist prioritization policy should be used.")
        ]="false",
         arrival_probabilities: Annotated[
             List[float], 
             typer.Option(help="Class arrival probabilities.")
        ]=[1/3, 1/3, 1/3],
         folder: Annotated[
             str, 
             typer.Option(help="Simulation results folder.")
        ]="./test/data",
         output_folder: Annotated[
             str, 
             typer.Option(help="Output folder for summary file.")
        ]="./test_data/test_summary/",
         virtual_att_probs: Annotated[
             Tuple[float, float, float, float],
             typer.Option(help="Virtual appointment attendance probabilities.")
         ]=(0.9,0.025,0.025,0.05),
         face_att_probs: Annotated[
             Tuple[float, float, float, float],
             typer.Option(help="In-person appointment attendance probabilities.")
         ]=(0.8,0.05,0.05,0.1)):
    
    # folder = folder + f"{directory}/" + sim_name + "/"
    # define path to C executable

    # define the program call
    program_call = [program_path]
    program_call += [f"--runs={runs}", f"--n_epochs={epochs}", f"--clinicians={clinicians}"]
    program_call += [f"--max_caseload={max_caseload}", f"--arr_lam={arr_lam}"]
    program_call += [f"--pathways={','.join(map(str, pathways))}", f"--wait_effects={','.join(map(str, wait_effects))}"]
    program_call += [f"--modality_effects={','.join(map(str, modality_effects))}", f"--modality_policies={','.join(map(str, modality_policies))}"]
    program_call += [f"--max_ax_age={max_ax_age}", f"--age_params={','.join(map(str, age_params))}"]
    program_call += [f"--priority_order={','.join(map(str, priority_order))}", f"--priority_wlist={priority_wlist}"]
    program_call += [f"--arrival_probs={','.join(map(str, arrival_probabilities))}"]
    program_call += [f"--virtual_att_probs={','.join(map(str,virtual_att_probs))}", f"--face_att_probs={','.join(map(str,face_att_probs))}"]
    program_call += [f"--folder={folder}"]
    print(program_call)
    
    # call the program as a subprocess and wait
    p1 = subprocess.Popen(program_call)
    exit_codes = p1.wait()

    # create summary file
    temp = pd.DataFrame()
    for i in range(runs):
        df = pd.read_parquet(folder + f"simulation_data_{i}.parquet")
        temp = pd.concat([temp, get_summary(df)], axis=0)
        os.remove(folder + f"simulation_data_{i}.parquet")   # remove the file after reading

    summary = get_mean_summary(temp, runs, arr_lam, arrival_probabilities, pathways, wait_effects, modality_effects, modality_policies,
                               virtual_att_probs, face_att_probs)
    del temp
    summary.to_parquet(output_folder + f"summary_{sim_name}.parquet")
    exit()

if __name__=="__main__":
    typer.run(main)