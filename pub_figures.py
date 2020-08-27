import os
import copy

import numpy
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats


# # # # #
# CONSTANTS

# Simulations to load data from.
NEUTRAL_SCENARIO = "neutral"
SCENARIOS = ["childenv", "childshaenv", "childnonenv", "parentenv", "noise"]
SCENARIOS.reverse()
# Proportion of variance due to confound in each scenario.
P_PER_SCENARIO = [10, 30, 50, 70, 90]
# Number of simulations per scenario.
N_SIMS = 66
# Number of runs per simulation.
N_RUNS = 100

# Files and folders
DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, "data")
OUTDIR = os.path.join(DIR, "pub_figures")
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

# Plot specs.
PLOT_CI_ALPHA = 0.5
FIGDPI = 300
PLOT_LINESTYLE = {"her":"-", "sha":"-", "non":"-"}
PLOT_LABELS = { \
    "childenv":     "Child impacts all environment", \
    "childshaenv":  "Child impacts shared environment", \
    "childnonenv":  "Child impacts non-shared environment", \
    "parentenv":    "Parent impacts shared environment", \
    "noise":        "Measurement error", \
    "her":          "Heritability", \
    "sha":          "Shared environment", \
    "non":          "Non-shared environment", \
    "tra": "Traditional estimation", \
    "sem": "Structural equation modelling", \
    }

PLOT_COLS = { \
    "childenv":     "#c4a000", \
    "childshaenv":  "#a40000", \
    "childnonenv":  "#ce5c00", \
    "parentenv":    "#204a87", \
    "noise":        "#4e9a06", \
    }


# # # # #
# LOAD DATA

# Start with an empty dict.
data = {}

# Loop through all scenarios.
for si, scenario in enumerate([NEUTRAL_SCENARIO] + SCENARIOS):
    
    if scenario == NEUTRAL_SCENARIO:
        p_in_this_scenario = [0]
        shape = (1, N_SIMS, N_RUNS)
    else:
        p_in_this_scenario = P_PER_SCENARIO
        shape = (1+len(P_PER_SCENARIO), N_SIMS, N_RUNS)

    # Create an empty dict for this scenario.
    data[scenario] = {}
    # Add a discrepancy dict for each twin-study estimation.
    data[scenario]["dis"] = {}
    data[scenario]["dis_p"] = {}
    data[scenario]["dis_con"] = {}
    for method in ["tra", "sem"]:
        data[scenario]["dis"][method] = {}
        data[scenario]["dis_p"][method] = {}
        data[scenario]["dis_con"][method] = {}
        for var in ["her", "sha", "non"]:
            data[scenario]["dis"][method][var] = numpy.zeros(shape, \
                dtype=float) * numpy.NaN
            data[scenario]["dis_p"][method][var] = numpy.zeros(shape, \
                dtype=float) * numpy.NaN
            data[scenario]["dis_con"][method][var] = numpy.zeros(shape, \
                dtype=float) * numpy.NaN
    
    # Loop through all simulations files within each scenario.
    for psi, p_scenario in enumerate(P_PER_SCENARIO):
        
        # Create an empty dict for this sub-scenario.
        data[scenario][p_scenario] = {}

        # Construct the path to this instances folder.
        scen_name = "parentenv-0_childshaenv-0_childnonenv-0_noise-0_simvar-1"
        if scenario == "neutral":
            pass
        elif scenario == "childenv":
            scen_name = scen_name.replace("childshaenv-0", \
                "childshaenv-{}".format(p_scenario))
            scen_name = scen_name.replace("childnonenv-0", \
                "childnonenv-{}".format(p_scenario))
        else:
            scen_name = scen_name.replace("{}-0".format(scenario), \
                "{}-{}".format(scenario, p_scenario))
        data_dir = os.path.join(DATADIR, scenario, scen_name)
        
        # Get all data files.
        all_fnames = os.listdir(data_dir)
        all_fnames.sort()
        
        # Loop through all files.
        for fi, fname in enumerate(all_fnames):
            
            # Split name and file extension.
            name, ext = os.path.splitext(fname)
            
            # Load data.
            raw = numpy.loadtxt(os.path.join(data_dir, fname), \
                delimiter=",", dtype=str, unpack=True)
            d = {}
            for i in range(raw.shape[0]):
                var = raw[i,0]
                val = raw[i,1:].astype(float)
                d[var] = val

            # Grab the data we need.
            for var in data.keys():
                data[scenario][p_scenario][name] = copy.deepcopy(d)
            
            # Compute the discrepancies.
            for method in ["tra", "sem"]:
                for var in ["her", "sha", "non"]:
                    if scenario == NEUTRAL_SCENARIO:
                        i = 0
                    else:
                        i = 1+psi
                    # Compute the discrepancy as the estimated minus the real
                    # proportion of variance.
                    data[scenario]["dis"][method][var][i,fi,:] = \
                        d["{}_{}".format(method, var)] - d["p_{}".format(var)]
                    # Compute the discrepancy if counting the confound.
                    if var == "her":
                        # Count the genetic confound in shared environment as
                        # genetical contribution.
                        real_val = d["p_her"] \
                            + d["p_child_shared_env"] * d["p_sha"] \
                            + d["p_child_nonshared_env"] * d["p_non"]
                    elif var == "sha":
                        # Count the genetic confound in shared environment as
                        # genetical contribution, so subtract it from the
                        # shared environment. This includes both child and
                        # parental genetic confounds.
                        real_val = d["p_sha"] \
                            - d["p_child_shared_env"] * d["p_sha"]
                    elif var == "non":
                        # Count the genetic confound in non-shared environment
                        # as genetical contribution, so subtract it from the
                        # non-shared environment.
                        real_val = d["p_non"] \
                            - d["p_child_nonshared_env"] * d["p_non"]
                    data[scenario]["dis_con"][method][var][i,fi,:] = \
                        d["{}_{}".format(method, var)] - real_val
                    # Compute the discrepancy as a proportion of the real
                    # proportion (only possible if p>0).
                    sel = d["p_{}".format(var)] > 0
                    data[scenario]["dis_p"][method][var][i,fi,sel] = \
                        d["{}_{}".format(method, var)][sel] \
                        / d["p_{}".format(var)][sel]

# Add neutral data to all non-neutral scenarios, because this is the null 
# estimate.
for scenario in SCENARIOS:
    for method in data[scenario]["dis"].keys():
        for var in data[scenario]["dis"][method].keys():
            data[scenario]["dis"][method][var][0,:,:] = \
                data[NEUTRAL_SCENARIO]["dis"][method][var][0,:,:]
            data[scenario]["dis_p"][method][var][0,:,:] = \
                data[NEUTRAL_SCENARIO]["dis_p"][method][var][0,:,:]
            data[scenario]["dis_con"][method][var][0,:,:] = \
                data[NEUTRAL_SCENARIO]["dis_con"][method][var][0,:,:]

# Extract discrepancy values and sort them.
fs = {}
for dis_var in ["dis", "dis_p", "dis_con"]:
    fs[dis_var] = {}
    for method in ["tra", "sem"]:
        fs[dis_var][method] = {}
        for scenario in SCENARIOS:
            fs[dis_var][method][scenario] = {}
            for var in ["her", "sha", "non"]:
                fs[dis_var][method][scenario][var] = []
                for i in range(data[scenario][dis_var][method][var].shape[0]):
                    # Flatten and sort data.
                    newshape = data[scenario][dis_var][method][var].shape[1] \
                        * data[scenario][dis_var][method][var].shape[2]
                    fs[dis_var][method][scenario][var].append( \
                        numpy.sort(numpy.reshape( \
                        data[scenario][dis_var][method][var][i,:,:], \
                        newshape)))
                    fs[dis_var][method][scenario][var][-1] = \
                        fs[dis_var][method][scenario][var][-1][numpy.isnan( \
                        fs[dis_var][method][scenario][var][-1])==False]

# Open a file to store stats output.
with open(os.path.join(OUTDIR, "stats.tsv"), "w") as f:
    # Write header.
    f.write("\t".join(["var", "n", "m", "sd", "sem", "z", "z_p", "t", "t_p", \
        "d", "str"]))

    # Compute z scores and effect sizes.
    z = {}
    d = {}
    for dis_var in ["dis", "dis_con"]:
        z[dis_var] = {}
        d[dis_var] = {}
        for method in ["tra", "sem"]:
            z[dis_var][method] = {}
            d[dis_var][method] = {}
            for scenario in SCENARIOS:
                z[dis_var][method][scenario] = {}
                d[dis_var][method][scenario] = {}
                for var in ["her", "sha", "non"]:
                    z[dis_var][method][scenario][var] = numpy.zeros( \
                        data[scenario][dis_var][method][var].shape[0], \
                        dtype=float) * numpy.NaN
                    d[dis_var][method][scenario][var] = numpy.zeros( \
                        data[scenario][dis_var][method][var].shape[0], \
                        dtype=float) * numpy.NaN
                    # Use a mean of 0 to compare against.
                    m0 = 0.0 
                    # If you're interested only in change from neutral:
                    #numpy.mean(fs[dis_var][method][scenario][var][0])
                    for i in range(data[scenario][dis_var][method][var].shape[0]):
                        # Compute mean and standard error.
                        n = fs[dis_var][method][scenario][var][i].shape[0]
                        m = numpy.mean(fs[dis_var][method][scenario][var][i])
                        sd = numpy.std(fs[dis_var][method][scenario][var][i])
                        sem = sd / numpy.sqrt(n)
                        z[dis_var][method][scenario][var][i] = (m-m0) / sem
                        zp = 2 * (1.0 - scipy.stats.norm.cdf(abs( \
                            z[dis_var][method][scenario][var][i])))
                        d[dis_var][method][scenario][var][i] = (m-m0) / sd
                        t, tp = scipy.stats.ttest_1samp( \
                            fs[dis_var][method][scenario][var][i], m0)
                        # Write to file.
                        if i == 0:
                            p_ = 0
                        else:
                            p_ = P_PER_SCENARIO[i-1]
                        res_str = "t({})={}, p={}".format(int(n-1), \
                            round(t,2), round(tp,3))
                        if tp < 0.05:
                            res_str += ", d={}".format(round( \
                                d[dis_var][method][scenario][var][i], 2))
                        line = [ \
                            "{}-{}-{}-{}-{}".format( \
                                dis_var, method, scenario, var, p_), \
                            n, m, sd, sem, \
                            z[dis_var][method][scenario][var][i], zp, \
                            t, tp, d[dis_var][method][scenario][var][i], \
                            res_str]
                        f.write("\n" + "\t".join(map(str, line)))


# # # # #
# FIGURE 1

# This figure show discrepancies between the real and estimated heritability,
# shared and non-shared environments against the ground truth.
for dis_var in ["dis", "dis_p", "dis_con"]:
    for method in ["tra", "sem"]:
        # Create a new figure.
        fig, axes = pyplot.subplots(nrows=2, ncols=3, figsize=(22.0,12.0), \
            dpi=FIGDPI)
        fig.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.95, \
            hspace=0.1, wspace=0.1)
    
        # Loop through all measurements.
        for vi, var in enumerate(["her", "sha", "non"]):
            
            # Choose Z axis.
            z_ax = axes[1,vi]
            # Draw invisible lines to indicate data types.
            z_ax.plot([-2,-1], [-1000, -1000], ls="-", lw=3, color="#000000", \
                alpha=0.7, label="Discrepancy")
            z_ax.plot([-2,-1], [-1000, -1000], ls=":", lw=3, color="#000000", \
                alpha=0.7, label="Counting G*E as genetic")
            # Draw lines to indicate statistical significance values.
            alpha = 0.005
            upper = scipy.stats.norm.ppf(1.0-alpha/2.0)
            lower = scipy.stats.norm.ppf(alpha/2.0)
            z_ax.axhline(y=upper, lw=2, ls="--", color="#000000", alpha=0.5)
            z_ax.axhline(y=lower, lw=2, ls="--", color="#000000", alpha=0.5, \
                label=r"Non-significant ($\alpha=$"+str(alpha)+")")
            # Shade area of non-significant results.
            z_ax.fill_between([-1, 2], [lower,lower], [upper, upper], \
                color="#000000", alpha=0.3)
            # Shade area of underestimation.
            z_ax.fill_between([-1, 2], [-1000,-1000], [lower, lower], \
                color="#000000", alpha=0.1)
            
            # Choose the right axis to draw in.
            ax = axes[0,vi]
            # Set axis title.
            ax.set_title(PLOT_LABELS[var], fontsize=28)
            
            # Clarify what is over/under estimation.
            if dis_var == "dis":
                mid_y = 0.0
                anno_y = [0.67, -0.55]
            elif dis_var == "dis_p":
                mid_y = 1.0
                anno_y = [3.7, 0.1]
            elif dis_var == "dis_con":
                mid_y = 0.0
                anno_y = [0.87, -0.45]
            # Draw line to split axis along y==0.
            ax.axhline(y=mid_y, ls="--", lw=2, color="#000000", alpha=0.3)
            # Shade area of underestimation.
            ax.fill_between([-1, 2], [-1,-1], [mid_y,mid_y], \
                color="#000000", alpha=0.1)
            # Annotate the over/under-estimation.
            if vi == 0:
                ax.annotate("Over-estimation", (-0.02, anno_y[0]), \
                    fontsize=18, color="#000000", alpha=0.5)
                ax.annotate("Under-estimation", (-0.02, anno_y[1]), \
                    fontsize=18, color="#000000", alpha=0.5)
            
            # Compute the x values, the proportion of confounding.
            p_confound = numpy.array([0] + P_PER_SCENARIO, dtype=float) / 100.0
    
            # Loop through all scenarios.
            for si, scenario in enumerate(SCENARIOS):
    
                # Compute the average discrepancy per proportion of confounding.
                m = numpy.nanmean(numpy.nanmedian( \
                    data[scenario][dis_var][method][var], axis=1), axis=1)
                sd = numpy.nanstd(numpy.nanmean( \
                    data[scenario][dis_var][method][var], axis=1), axis=1)
                # Z values for corrected and uncorrected discrepnancies.
                zu = numpy.zeros(m.shape[0], dtype=float)
                zc = numpy.zeros(m.shape[0], dtype=float)
                # Compute the confidence intervals.
                alpha_range = numpy.arange(0.5, 1.01, 0.02)
                alpha_step = 0.3 / alpha_range.shape[0]
                for PLOT_CI_ALPHA in alpha_range:
                    ci_lo = numpy.zeros(m.shape)
                    ci_hi = numpy.zeros(m.shape)
                    for i in range(len(fs[dis_var][method][scenario][var])):
                        # Find the low and high interval bounds.
                        li = max(0, int(round( \
                            fs[dis_var][method][scenario][var][i].shape[0] \
                            * (PLOT_CI_ALPHA/2.0), 0)))
                        hi = min( \
                            fs[dis_var][method][scenario][var][i].shape[0]-1, \
                            int(fs[dis_var][method][scenario][var][i].shape[0] \
                            * (1.0-(PLOT_CI_ALPHA/2.0))))
                        ci_lo[i] = fs[dis_var][method][scenario][var][i][li]
                        ci_hi[i] = fs[dis_var][method][scenario][var][i][hi]
                    # Shade confidence interval.
                    ax.fill_between(p_confound, ci_lo, ci_hi, \
                        color=PLOT_COLS[scenario], alpha=alpha_step)
                
                # Plot line.
                ax.plot(p_confound, m, ls="-", lw=3, color=PLOT_COLS[scenario], \
                    label=PLOT_LABELS[scenario], alpha=0.7)
                # Plot z values.
                z_ax.plot(p_confound, z["dis"][method][scenario][var], \
                    lw=3, ls="-", color=PLOT_COLS[scenario], alpha=0.7)
                z_ax.plot(p_confound, z["dis_con"][method][scenario][var], \
                    lw=3, ls=":", color=PLOT_COLS[scenario], alpha=0.7)
            
            # Set axis limits.
            ax.set_xlim(-0.05, 1)
            z_ax.set_xlim(-0.05, 1)
            z_ax.set_ylim(-150, 220)
            if dis_var == "dis":
                ax.set_ylim(-0.6, 0.8)
            elif dis_var == "dis_p":
                ax.set_ylim(0, 4)
            elif dis_var == "dis_con":
                ax.set_ylim(-0.5, 1)
            # Set axis ticks.
            xticks = numpy.round(numpy.arange(0, 1.01, 0.1), 1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([])
            z_ax.set_xticks(xticks)
            z_ax.set_xticklabels(xticks, fontsize=14)
            if dis_var == "dis":
                yticks = numpy.round(numpy.arange(-0.6, 0.81, 0.2), 1)
            elif dis_var == "dis_p":
                yticks = numpy.round(numpy.arange(0, 4.01, 0.5), 1)
            elif dis_var == "dis_con":
                yticks = numpy.round(numpy.arange(-0.4, 1.01, 0.2), 1)
            zyticks = numpy.round(numpy.arange(-150, 201, 50), 0)
            ax.set_yticks(yticks)
            z_ax.set_yticks(zyticks)
            if vi == 0:
                ax.set_yticklabels(yticks, fontsize=14)
                z_ax.set_yticklabels(zyticks, fontsize=14)
            else:
                ax.set_yticklabels([])
                z_ax.set_yticklabels([])
            # Set axis labels.
            z_ax.set_xlabel("Proportion of confounding", fontsize=24)
            if vi == 0:
                ax.set_ylabel("Estimation error", fontsize=24)
                z_ax.set_ylabel("Z statistic", fontsize=24)
        
        # Add a legend to the middle axis.
        axes[0,1].legend(loc="upper left", fontsize=16)
        axes[1,1].legend(loc="upper left", fontsize=16)
        # Save figure.
        fig.savefig(os.path.join(OUTDIR, "fig-01_discrepancies_{}_{}.jpg".format( \
            dis_var, method)))
        pyplot.close(fig)


# # # # #
# REST OF THE FIGURES

fig_details = { \
    "fig-s02_neutral": { \
        "scenario": "neutral", \
        "sim_name": "parentenv-0_childshaenv-0_childnonenv-0_noise-0_simvar-1", \
        }, \
    "fig-02_childenv": { \
        "scenario": "childenv", \
        "sim_name": "parentenv-0_childshaenv-10_childnonenv-10_noise-0_simvar-1", \
        }, \
    "fig-03_parentenv": { \
        "scenario": "parentenv", \
        "sim_name": "parentenv-30_childshaenv-0_childnonenv-0_noise-0_simvar-1", \
        }, \
    "fig-04_noise": { \
        "scenario": "noise", \
        "sim_name": "parentenv-0_childshaenv-0_childnonenv-0_noise-30_simvar-1", \
        }, \
    "fig-s03_child_shared_env": { \
        "scenario": "childshaenv", \
        "sim_name": "parentenv-0_childshaenv-10_childnonenv-0_noise-0_simvar-1", \
        }, \
    "fig-s04_child_nonshared_env": { \
        "scenario": "childnonenv", \
        "sim_name": "parentenv-0_childshaenv-0_childnonenv-10_noise-0_simvar-1", \
        }, \
    }

# Choose what values to plot.
p_her_plot = [0.1, 0.3, 0.5, 0.7, 0.9]
p_sha_plot = numpy.arange(0.0, 1.01, 0.1)
p_non_plot = numpy.arange(0.0, 1.01, 0.1)

# Create a normed colour map to indicate years. Usage: cmap(norm(p))
cmap = matplotlib.cm.get_cmap("viridis")
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

# Loop through all figures.
for fig_name in fig_details.keys():
    
    # Get the simulation specifics.
    scenario = fig_details[fig_name]["scenario"]
    sim_name = fig_details[fig_name]["sim_name"]

    # Create a new figure.
    fig, axes = pyplot.subplots(nrows=1, ncols=3, figsize=(24.0, 7.0), \
        dpi=FIGDPI)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.11, top=0.93, \
        wspace=0.15, hspace=0.2)
    
    # Plot all true values.
    for vi, var in enumerate(["her", "sha", "non"]):
        if var == "her":
            for p_her in p_her_plot:
                axes[vi].plot([-1,2], [p_her, p_her], ls="--", lw=2, \
                    color=cmap(norm(p_her)), alpha=0.5)
        if var == "sha":
            axes[vi].plot([-1,2], [-1, 2], ls="--", lw=2, \
                color="#000000", alpha=0.3, label="Ground truth")
        if var == "non":
            for p_her in p_her_plot:
                p_non = 1.0 - p_her - p_sha_plot
                axes[vi].plot(p_sha_plot, p_non, ls="--", lw=2, \
                    color=cmap(norm(p_her)), alpha=0.5)

    # Loop through both estimation methods.
    method = "tra"
    # Set the N to plot.
    n_twins = 1000
    n_runs = 100
    # Run through all simulations.
    for pi, p_her in enumerate(p_her_plot):
        
        # Create empty data array. A large part of this will remain empty.
        shape = (p_sha_plot.shape[0], n_runs)
        data = { \
            "her":numpy.zeros(shape, dtype=float) * numpy.NaN, \
            "sha":numpy.zeros(shape, dtype=float) * numpy.NaN, \
            "non":numpy.zeros(shape, dtype=float) * numpy.NaN, \
            "p_child_shared_env":0, \
            "p_child_nonshared_env":0, \
            }
    
        # Load all files.
        for psi, p_sha in enumerate(p_sha_plot):
            # Construct the path to the data file.
            p_non = 1.0 - p_her - p_sha
            fpath = os.path.join(DATADIR, scenario, sim_name, \
                "n-{}_her-{}_sha-{}_non-{}.csv".format(n_twins, \
                int(round(p_her*100)), int(round(p_sha*100)), \
                int(round(p_non*100))))

            # Load the data.
            if os.path.isfile(fpath):
                raw = numpy.loadtxt(fpath, delimiter=",", dtype=str, \
                    unpack=True)
                d = {}
                for i in range(raw.shape[0]):
                    var = raw[i,0]
                    val = raw[i,1:].astype(float)
                    d[var] = val
                # Grab the data we need.
                for var in ["her", "sha", "non"]:
                    data[var][psi,:] = d["{}_{}".format(method,var)]
                data["p_child_shared_env"] = d["p_child_shared_env"][0]
                data["p_child_nonshared_env"] = d["p_child_nonshared_env"][0]
        
        # Plot all lines.
        for vi, var in enumerate(["her", "sha", "non"]):

            # Plot the "real" values if child environment is confounded
            # with child genotype.
            if (data["p_child_shared_env"] > 0) or (data["p_child_nonshared_env"] > 0):
                if var == "her":
                    p_her_real = p_her + \
                        p_sha_plot*data["p_child_shared_env"] + \
                        (1.0-p_her-p_sha_plot)*data["p_child_nonshared_env"]
                    axes[vi].plot(p_sha_plot, p_her_real, \
                        ls=":", lw=2, color=cmap(norm(p_her)), alpha=0.5)
                if var == "sha" and pi == 0:
                    p_sha_plot_real = p_sha_plot - \
                        (p_sha_plot*data["p_child_shared_env"])
                    axes[vi].plot(p_sha_plot, p_sha_plot_real, \
                        ls=":", lw=2, color="#000000", alpha=0.3, \
                        label="Counting G*E as genetic")
                if var == "non":
                    p_non = 1.0 - p_her - p_sha_plot
                    p_non_real = p_non - p_non*data["p_child_nonshared_env"]
                    axes[vi].plot(p_sha_plot, p_non_real, \
                        ls=":", lw=2, color=cmap(norm(p_her)), alpha=0.5)

            # Compute average and confidence 90% intervals.
            m = numpy.nanmean(data[var], axis=1)
            s = numpy.sort(data[var], axis=1)
            ci5 = s[:,int(round(s.shape[1]*0.05))]
            ci95 = s[:,int(round(s.shape[1]*0.95))]
            # Plot averages and confidence intervals.
            axes[vi].plot(p_sha_plot, m, ls=PLOT_LINESTYLE[var], lw=3, \
                color=cmap(norm(p_her)), alpha=0.5)
            axes[vi].fill_between(p_sha_plot, ci5, ci95, \
                color=cmap(norm(p_her)), alpha=0.3)

            # Finish the plot in the first run-through.
            if pi == 0:
                # Set colour bar ticks.
                bticks = numpy.arange(0, 1.01, 0.2)
                # Add the colour bar.
                divider = make_axes_locatable(axes[vi])
                bax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = matplotlib.colorbar.ColorbarBase(bax, cmap=cmap, norm=norm, \
                    ticks=bticks, orientation='vertical')
                cbar.set_ticklabels(numpy.round(bticks,1))
                cbar.ax.tick_params(labelsize=15)
                if vi == axes.shape[0]-1:
                    cbar.set_label("Simulated heritability", fontsize=26)
            
                # Set axis ticks.
                ticks = numpy.arange(0.0, 1.01, 0.2)
                axes[vi].set_xticks(ticks)
                axes[vi].set_xticklabels(numpy.round(ticks,1), fontsize=17)
                axes[vi].set_yticks(ticks)
                if vi == 0:
                    axes[vi].set_yticklabels(numpy.round(ticks,1), fontsize=17)
                else:
                    axes[vi].set_yticklabels([])
                # Set axis limits.
                axes[vi].set_xlim(0, 1)
                axes[vi].set_ylim(0, 1)
                # Set axis labels.
                axes[vi].set_title(PLOT_LABELS[var], fontsize=26)
                axes[vi].set_xlabel("Simulated {}".format( \
                    PLOT_LABELS["sha"].lower()), fontsize=26)
                if vi == 0:
                    axes[vi].set_ylabel("Estimated proportion", fontsize=26)
    
    # Draw legend for the middle panel.
    axes[1].legend(loc="upper left", fontsize=17)
    # Save the figure.
    outpath = os.path.join(OUTDIR, "{}.jpg".format(fig_name))
    fig.savefig(outpath)
    pyplot.close(fig)

