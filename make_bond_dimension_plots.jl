include("plot_data.jl")

plt.rc("figure", figsize=[8,5])
# plt.rc("legend", loc="lower right")
plt.rc("font", size=12)
colormap = plt.get_cmap("inferno")
interval_low = 0.0
interval_high = 0.9
dpi = 300
format = "png"

############# SU(2) point #############

# Rung perturbations
perts = 0.1:0.1:0.5
fig,ax = plt.subplots(2, sharey="row", gridspec_kw=Dict("height_ratios" => [3,2]), layout="constrained")
# plt.tight_layout()
ax[1].set_xlim(8e-1, 200)
ax[1].set_ylim(5e-4, 1e-3)
ax[1].set_prop_cycle(plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts)+1)]))
ax[2].set_prop_cycle(plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts)+1)]))

# plot_hdf(ax, "data_plots/tdvp_coarsegrained_dw_gpu_L128_chi512_beta0.0_dt0.1_Jprime0.0_mu0.001.h5", type="hdf", graph="both_transfer", t_scale=0.0)
# plot_hdf(ax, "data_plots/production/tdvp_coarsegrained_dw_gpu_L128_chi512_beta0.0_dt0.1_Jprime0.0_U0.0_Uprime0.0_mu0.001.h5", type="hdf", graph="both_transfer", t_scale=1.0, T_cutoff=80.0)
# plot_hdf(ax, "data_plots/production/tdvp_coarsegrained_dw_L64_chi1024_beta0.0_dt_ramped0.1_20.0_0.5_Jprime0.0_U0.0_Uprime0.0_mu0.001_1e12cutoff.h5", type="hdf", graph="both_transfer", t_scale=1.0)
plot_hdf(ax, "data_plots/production/chain_L400_chi400_mu0.0017_1e12.h5", type="hdf", graph="both_transfer", t_scale=0.0, s_scale=1.17, T_cutoff=100.0, dw="su(3)")
for J2 in perts
    # file = "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime$(J2)_mu0.001.h5"
    # file = "data_plots/production/tdvp_coarsegrained_dw_gpu_L64_chi600_beta0.0_dt0.1_Jprime$(J2)_U0.0_Uprime0.0_mu0.001.h5"
    file = "data_plots/production/tdvp_coarsegrained_dw_gpu_L128_chi512_beta0.0_dt0.1_Jprime$(J2)_U0.0_Uprime0.0_mu0.001.h5"
    # file = "data_plots/production/tdvp_coarsegrained_dw_L64_chi1024_beta0.0_dt_ramped0.1_20.0_0.5_Jprime$(J2)_U0.0_Uprime0.0_mu0.001_1e12cutoff.h5"
    plot_hdf(ax, file, type="hdf", graph="both_transfer", label="J'=$(J2)", t_scale=J2^2, window_size=30, T_cutoff=100.0)
end
# axs[1].set_title("Magnetization transfer from initial domain wall (J' perturbations around U=0.0)")
ax[1].set_xlabel(latexstring("t"))
ax[1].set_ylabel(latexstring("\\Delta s \\cdot t^{-2/3}"))
ax[2].set_xlabel(latexstring("t \\cdot J'^2"))
ax[2].set_ylabel(latexstring("z"))
ax[1].set_title("L=128, χ=512")
# plt.savefig("plots/su2_J.$(format)", dpi=dpi)

norm = plt.Normalize(vmin=0.0, vmax=0.5)
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
cbar = plt.colorbar(sm, ax=ax[1], pad=0.03, ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
cbar.ax.set_xlabel(latexstring("J'"))

plt.savefig("plots/su2_J_L128_chi512.$(format)", dpi=dpi)
