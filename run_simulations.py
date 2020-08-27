import os
import sys
import copy
import time
import itertools

import numpy

from simtwin import twin_sim


# Skip simulations for which an output file exists.
SKIP_EXISTING = False

# Files and folders.
DIR = os.path.abspath(os.path.dirname(__file__))
DATADIR = os.path.join(DIR, "data")
if not os.path.isdir(DATADIR):
    os.mkdir(DATADIR)

# Define the number of twins per study, and the MZ proportion.
p_mz_twins = 0.5
n_twins = 1000
# Define the number of simulations per unique combination.
n_runs = 100
# Set default scenario options.
sim_var = True
p_parent_env = [0]
p_child_shared_env = [0]
p_child_nonshared_env = [0]
p_noise = [0]
default_props = [0.1, 0.3, 0.5, 0.7, 0.9]

# Check if an argument was passed. If it wasn't, assume default values.
scenario = "neutral"
if len(sys.argv) > 1:
    scenario = sys.argv[1]
# Allow default numbers of runs and simulated twins to be overwritten.
if len(sys.argv) > 2:
    n_runs = int(sys.argv[2])
if len(sys.argv) > 3:
    n_twins = int(sys.argv[3])

# Apply scenario settings.
if scenario == "neutral":
    sim_var = True
    p_parent_env = [0]
    p_child_shared_env = [0]
    p_child_nonshared_env = [0]
    p_noise = [0]
    
elif scenario == "parentenv":
    sim_var = True
    p_parent_env = copy.deepcopy(default_props)
    p_child_shared_env = [0]
    p_child_nonshared_env = [0]
    p_noise = [0]

elif scenario == "childenv":
    sim_var = True
    p_parent_env = [0]
    p_child_shared_env = copy.deepcopy(default_props)
    p_child_nonshared_env = copy.deepcopy(default_props)
    p_noise = [0]

elif scenario == "childshaenv":
    sim_var = True
    p_parent_env = [0]
    p_child_shared_env = copy.deepcopy(default_props)
    p_child_nonshared_env = [0]
    p_noise = [0]

elif scenario == "childnonenv":
    sim_var = True
    p_parent_env = [0]
    p_child_shared_env = [0]
    p_child_nonshared_env = copy.deepcopy(default_props)
    p_noise = [0]
    
elif scenario == "noise":
    sim_var = True
    p_parent_env = [0]
    p_child_shared_env = [0]
    p_child_nonshared_env = [0]
    p_noise = copy.deepcopy(default_props)

# If specific options were passed, parse them individually.
else:
    if "parentenv" in scenario:
        si = scenario.find("parentenv-")+10
        p_parent_env = [float(scenario[si:si+2])/100.0]
    if "childenv" in scenario:
        si = scenario.find("childenv-")+9
        p_child_shared_env = [float(scenario[si:si+2])/100.0]
        p_child_nonshared_env = [float(scenario[si:si+2])/100.0]
    if "childshaenv" in scenario:
        si = scenario.find("childshaenv-")+12
        p_child_shared_env = [float(scenario[si:si+2])/100.0]
    if "childnonenv" in scenario:
        si = scenario.find("childnonenv-")+12
        p_child_nonshared_env = [float(scenario[si:si+2])/100.0]
    if "noise" in scenario:
        si = scenario.find("noise-")+6
        p_noise = [float(scenario[si:si+2])/100.0]
    if "simvar" in scenario:
        si = scenario.find("simvar-")+7
        simvar = scenario[si:si+1] == "1"

# Create a list of all scenarios to run through.
all_run_specs = []
for p_parent_env_ in p_parent_env:
    for p_child_shared_env_ in p_child_shared_env:
        for p_child_nonshared_env_ in p_child_nonshared_env:
            # In the combined scenario, we only want matching shared and 
            # non-shared environments.
            if (scenario == "childenv") and \
                (p_child_shared_env_ != p_child_nonshared_env_):
                continue
            for p_noise_ in p_noise:
                all_run_specs.append(( \
                    copy.copy(p_parent_env_), \
                    copy.copy(p_child_shared_env_), \
                    copy.copy(p_child_nonshared_env_), \
                    copy.copy(p_noise_)))
# Unreference variables to avoid later confusion.
del p_parent_env, p_child_shared_env, p_child_nonshared_env, p_noise

# Define proportions of heritability and shared environment that should be
# tested.
p_her = numpy.arange(0.0, 1.01, 0.1)
p_sha = numpy.arange(0.0, 1.01, 0.1)
# Create a list of all unique and viable combinations.
all_p_combs = []
for ph in p_her:
    for ps in p_sha:
        # Only use combinations that add up to 1.
        if ph+ps <= 1:
            # Compute the non-shared environment from the genetic and shared
            # environment. Use abs to avoid -0.
            pu = abs(1.0 - ph - ps)
            # Add the combination to the list of unique combinations. The
            # rounding is to get rid of float artifacts.
            all_p_combs.append((round(ph,2), round(ps,2), round(pu,2)))

# Overwrite data directory with scenario name.
DATADIR = os.path.join(DATADIR, scenario)
if not os.path.isdir(DATADIR):
    os.mkdir(DATADIR)

# Define the variables that should be logged, on top of the simulation 
# specifications that will also be logged.
log_vars = ["r_mz", "r_dz", "tra_her", "tra_sha", "tra_non", \
    "sem_her", "sem_sha", "sem_non", "r_error"]

# Run through all simulations.
for si, (p_parent_env, p_child_shared_env, p_child_nonshared_env, p_noise) \
    in enumerate(all_run_specs):
    
    # Construct an output directory for this specific simulation.
    name = "parentenv-{}_childshaenv-{}_childnonenv-{}_noise-{}_simvar-{}"
    name = name.format(int(100*p_parent_env), int(100*p_child_shared_env), \
        int(100*p_child_nonshared_env), int(100*p_noise), int(sim_var))
    data_dir = os.path.join(DATADIR, name)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # Run through all combinations of heritability and environments.
    for pi, (p_her, p_sha, p_non) in enumerate(all_p_combs):
    
        # Construct the path to the output file.
        fpath = os.path.join(data_dir, "n-{}_her-{}_sha-{}_non-{}.csv".format( \
            n_twins, int(round(p_her*100)), int(round(p_sha*100)), \
            int(round(p_non*100))))
        # Create a new file if one does not exist yet.
        if not os.path.isfile(fpath):
            with open(fpath, "w") as f:
                header = ["n", "sim_var", "p_her", "p_sha", "p_non", \
                    "p_parent_env", "p_child_shared_env", \
                    "p_child_nonshared_env", "p_noise"] + log_vars
                f.write(",".join(header))
        # Skip this simulation if we're skipping existing files.
        else:
            if SKIP_EXISTING:
                continue
    
        # Run through all runs.
        print("Running {} simulations {}/{}".format(n_runs, \
            1+pi+si*len(all_p_combs), len(all_run_specs)*len(all_p_combs)))
        t0 = time.time()
        for i in range(n_runs):
            # Simulate the twin study.
            result = twin_sim(p_her, p_sha, p_non, sim_var=sim_var, \
                n_twins=n_twins, p_mz_twins=p_mz_twins, n_genes=10, \
                alleles=range(-7,8), p_alleles=None, env_m=0.0, env_sd=1.0, \
                p_parent_env=p_parent_env, p_child_shared_env=p_child_shared_env, \
                p_child_nonshared_env=p_child_nonshared_env, p_noise=p_noise, \
                noise_m=0.0, noise_sd=1.0)
            # Log the results.
            with open(fpath, "a") as f:
                line = [n_twins, int(sim_var), p_her, p_sha, p_non, p_parent_env, \
                    p_child_shared_env, p_child_nonshared_env, p_noise]
                line += [result[var] for var in log_vars]
                f.write("\n" + ",".join(map(str, line)))
        t1 = time.time()
        print("\tFinished in {} seconds".format(round(t1-t0,3)))

