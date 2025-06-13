include("plot_data.jl")

plt.rc("figure", figsize=[12,5])
# plt.rc("legend", loc="lower right")
plt.rc("font", size=12)
colormap = plt.get_cmap("inferno")
interval_low = 0.0
interval_high = 0.9
dpi = 1200
format = "png"

############# SU(2) point #############

# Rung perturbations
perts = 0.1:0.1:0.5
fig,axs = plt.subplots(2,3, sharey="row", gridspec_kw=Dict("height_ratios" => [3,2], "width_ratios" => [1, 1, 1]), layout="constrained")
# plt.tight_layout()
ax = axs[:,1]
ax[1].set_xlim(8e-1, 110)
ax[1].set_ylim(5e-4, 1e-3)
ax[1].set_prop_cycle(plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts)+1)]))
ax[2].set_prop_cycle(plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts)+1)]))

plot_hdf(ax, "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime0.0_mu0.001.h5", type="hdf", graph="both_transfer", t_scale=0.0)
for J2 in perts
    # file = "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime$(J2)_mu0.001.h5"
    file = "data_plots/production/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime$(J2)_U0.0_Uprime0.0_mu0.001.h5"
    # file = "data_plots/production/tdvp_coarsegrained_dw_L64_chi1024_beta0.0_dt_ramped0.1_20.0_0.5_Jprime$(J2)_U0.0_Uprime0.0_mu0.001_1e12cutoff.h5"
    plot_hdf(ax, file, type="hdf", graph="both_transfer", label="J'=$(J2)", t_scale=J2^2, window_size=30)
end
# axs[1].set_title("Magnetization transfer from initial domain wall (J' perturbations around U=0.0)")
ax[1].set_xlabel(latexstring("t"))
ax[1].set_ylabel(latexstring("\\Delta s \\cdot t^{-2/3}"))
ax[2].set_xlabel(latexstring("t \\cdot J'^2"))
ax[2].set_ylabel(latexstring("z"))
# plt.savefig("plots/su2_J.$(format)", dpi=dpi)

# Sym-breaking Biquad. perturbations
perts = 0.4:0.4:2.0
# fig,axs = plt.subplots(2,sharex=true)
ax = axs[:,2]
ax[1].set_xlim(8e-1, 100)
ax[1].set_prop_cycle(plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts)+1)]))
ax[2].set_prop_cycle(plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts)+1)]))

plot_hdf(ax, "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime0.0_mu0.001.h5", type="hdf", graph="both_transfer", t_scale=0.0)
for U in perts
    # file = "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime0.0_U0.0_Uprime$(U)_mu0.001.h5"
    file = "data_plots/production/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime0.0_U0.0_Uprime$(U)_mu0.001.h5"
    λ = U/(1+U)
    plot_hdf(ax, file, type="hdf", graph="both_transfer", label="U'=$(U/4)", t_scale=λ^2)
end
# axs[1].set_title("Magnetization transfer from initial domain wall (U' perturbations around U=0.0)")
ax[1].set_xlabel(latexstring("t"))
# ax[1].set_ylabel(latexstring("\\Delta s \\cdot t^{-2/3}"))
ax[2].set_xlabel(latexstring("t \\cdot U'^2"))
# ax[2].set_ylabel(latexstring("z"))
# plt.savefig("plots/su2_Uprime.$(format)", dpi=dpi)

# Sym-preserving Biquad. perturbations
perts = 0.0:0.4:2.0
# fig,axs = plt.subplots(2,sharex=true)
ax = axs[:,3]
ax[1].set_xlim(6e-1, 80)
ax[1].set_prop_cycle(plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))
ax[2].set_prop_cycle(plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))

# plot_hdf(ax, "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime0.0_mu0.001.h5", type="hdf", graph="both_transfer", t_scale=0.0)
for U in perts
    file = "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime0.0_U$(U)_mu0.001.h5"
    # file = "data_plots/production/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime0.0_U$(U)_Uprime0.0_mu0.001.h5"
    λ = U/(1+U)
    # plot_hdf(ax, file, type="hdf", graph="both_transfer", label="U=$(U/4)", t_scale=U^6)
    plot_hdf(ax, file, type="hdf", graph="both_transfer", label="U=$(U/4)", t_scale=λ^6/(1-λ), window_size=25)
end
# axs[1].set_title("Magnetization transfer from initial domain wall (U perturbations around U=0.0)")
ax[1].set_xlabel(latexstring("t"))
# axs[1].set_ylabel(latexstring("\\Delta s \\cdot t^{-2/3}"))
# ax[2].set_xlabel(latexstring("t \\cdot U^6"))
ax[2].set_xlabel(latexstring("\\frac{t \\cdot λ^6}{1-λ}"))
# axs[2].set_ylabel(latexstring("z"))

norm = plt.Normalize(vmin=0.0, vmax=0.5)
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
cbar = plt.colorbar(sm, ax=axs[1,1], pad=0.03, ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
cbar.ax.set_xlabel(latexstring("J'"))
cbar = plt.colorbar(sm, ax=axs[1,2], pad=0.02, ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
cbar.ax.set_xlabel(latexstring("U'"))
cbar = plt.colorbar(sm, ax=axs[1,3], pad=0.0, ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
cbar.ax.set_xlabel(latexstring("U"))

plt.savefig("plots/su2.$(format)", dpi=dpi)




############ SU(4) point ##############

# Rung perturbations
perts = 0.0:0.1:0.5
fig,axs = plt.subplots(2,3,sharey="row", gridspec_kw=Dict("height_ratios" => [3, 2]), layout="constrained")
ax = axs[:,1]
ax[1].set_xlim(2e-1, 80)
ax[1].set_ylim(8e-4, 1.8e-3)
ax[1].set_prop_cycle(plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))
ax[2].set_prop_cycle(plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))
for J2 in perts
    # file = "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime$(J2)_U4.0_mu0.001.h5"
    file = "data_plots/production/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime$(J2)_U4.0_Uprime0.0_mu0.001.h5"
    plot_hdf(ax, file, type="hdf", graph="both_transfer", label="J'=$(J2)")
end
# axs[1].set_title("Magnetization transfer from initial domain wall (J' perturbations around U=4.0)")
ax[1].set_xlabel(latexstring("t"))
ax[1].set_ylabel(latexstring("\\Delta s \\cdot t^{-2/3}"))
ax[2].set_xlabel(latexstring("t"))
ax[2].set_ylabel(latexstring("z"))
# plt.savefig("plots/su4_J.$(format)", dpi=dpi)

# Sym-breaking Biquad. perturbations
perts = 0.0:0.4:2.0
# fig,axs = plt.subplots(2,sharex=true)
ax = axs[:,2]
ax[1].set_xlim(2e-1, 80)
ax[1].set_prop_cycle(plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))
ax[2].set_prop_cycle(plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))
for U in perts
    # file = "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime0.0_U4.0_Uprime$(U)_mu0.001.h5"
    file = "data_plots/production/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime0.0_U4.0_Uprime$(U)_mu0.001.h5"
    plot_hdf(ax, file, type="hdf", graph="both_transfer", label="U'=$(U/4)", t_scale=U^2, window_size=30)
end
# axs[1].set_title("Magnetization transfer from initial domain wall (U' perturbations around U=4.0)")
ax[1].set_xlabel(latexstring("t"))
# ax[1].set_ylabel(latexstring("\\Delta s \\cdot t^{-2/3}"))
ax[2].set_xlabel(latexstring("t \\cdot U'^2"))
# ax[2].set_ylabel(latexstring("z"))
# plt.savefig("plots/su4_Uprime.$(format)", dpi=dpi)

# Sym-preserving Biquad. perturbations
# perts = [0.1,0.2,0.3,0.4,0.5,0.8,1.2,1.6,2.0]
perts = 0.0:0.4:2.0
# fig,axs = plt.subplots(2,sharex=true)
ax = axs[:,3]
ax[1].set_xlim(2e-1, 80)
ax[1].set_prop_cycle(plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))
ax[2].set_prop_cycle(plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))
for U in perts
    # file = "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime0.0_U$(U+4)_mu0.001.h5"
    file = "data_plots/production/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime0.0_U$(U+4)_Uprime0.0_mu0.001.h5"
    plot_hdf(ax, file, type="hdf", graph="both_transfer", label="δU=$(U/4)", t_scale=U^2, window_size=30)
end
# axs[1].set_title("Magnetization transfer from initial domain wall (U perturbations around U=4.0)")
ax[1].set_xlabel(latexstring("t"))
# ax[1].set_ylabel(latexstring("\\Delta s \\cdot t^{-2/3}"))
ax[2].set_xlabel(latexstring("t \\cdot δU^2"))
# ax[2].set_ylabel(latexstring("z"))

norm = plt.Normalize(vmin=0.0, vmax=0.5)
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
cbar = plt.colorbar(sm, ax=axs[1,1], pad=0.03, ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
cbar.ax.set_xlabel(latexstring("J'"))
cbar = plt.colorbar(sm, ax=axs[1,2], pad=0.02, ticks=[0.0, 0.1, 0.2, 0.3, 0.4])
cbar.ax.set_xlabel(latexstring("U'"))
cbar = plt.colorbar(sm, ax=axs[1,3], pad=0.0, ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
cbar.ax.set_xlabel(latexstring("\\delta U"))

plt.savefig("plots/su4.$(format)", dpi=dpi)

############ SU(3) ladder ##############

# Sym-preserving Biquad. perturbations
perts = [0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.4,1.6]
plt.rc("axes", prop_cycle=plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))
plt.rc("figure", figsize=[7,7])
plt.rc("font", size=15)
fig,axs = plt.subplots(2, gridspec_kw=Dict("height_ratios" => [3, 2]), layout="constrained")
for U in perts
    file = "data_plots/tebd_su(3)_dw_L64_chi512_beta0.0_dt0.1_U$(U)_mu0.001_conserve_threesite_conj.h5"
    λ = U/(1+U)
    plot_hdf(axs, file, type="hdf", graph="both_transfer", label="U=$U", t_scale=U^6, ncol=2, window_size=15)
    # plot_hdf(axs, file, type="hdf", graph="both_transfer", label="U=$U", t_scale=λ^6/(1-λ))
end
# axs[1].set_title("Magnetization transfer from initial domain wall (U perturbations)")
axs[1].set_xlim(3e-1, 30)
axs[1].set_ylim(2e-3, 5e-3)
axs[1].set_xlabel(latexstring("t"))
axs[1].set_ylabel(latexstring("\\Delta s \\cdot t^{-2/3}"))
axs[2].set_xlabel(latexstring("t \\cdot U^6"))
axs[2].set_ylabel(latexstring("z"))

norm = plt.Normalize(vmin=0.0, vmax=1.6)
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
cbar = plt.colorbar(sm, ax=axs[1,1], pad=0.03, ticks=[0.0, 0.4, 0.8, 1.2, 1.6])
cbar.ax.set_xlabel(latexstring("U"))

plt.savefig("plots/su3_U.$(format)", dpi=dpi)