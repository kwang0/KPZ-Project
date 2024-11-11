include("plot_data.jl")

plt.rc("figure", figsize=[6,6])
colormap = plt.get_cmap("inferno")
interval_low = 0.0
interval_high = 0.9
dpi = 1200
format = "pdf"

############# SU(2) point #############

# Rung perturbations
perts = 0.1:0.1:0.5
plt.rc("axes", prop_cycle=plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))
fig,axs = plt.subplots(2,sharex=true)
for J2 in perts
    file = "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime$(J2)_mu0.001.h5"
    plot_hdf(axs, file, type="hdf", graph="both_transfer", label="J'=$(J2)", t_scale=J2^2)
end
# axs[1].set_title("Magnetization transfer from initial domain wall (J' perturbations around U=0.0)")
axs[1].set_xlabel(latexstring("t"))
axs[1].set_ylabel(latexstring("\\Delta s \\cdot t^{-2/3}"))
axs[2].set_xlabel(latexstring("t \\cdot J'^2"))
axs[2].set_ylabel(latexstring("z"))
plt.savefig("plots/su2_J.$(format)", dpi=dpi)

# Sym-breaking Biquad. perturbations
perts = 0.4:0.4:2.0
plt.rc("axes", prop_cycle=plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))
fig,axs = plt.subplots(2,sharex=true)
for U in perts
    file = "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime0.0_U0.0_Uprime$(U)_mu0.001.h5"
    plot_hdf(axs, file, type="hdf", graph="both_transfer", label="U'=$(U)", t_scale=U^2)
end
# axs[1].set_title("Magnetization transfer from initial domain wall (U' perturbations around U=0.0)")
axs[1].set_xlabel(latexstring("t"))
axs[1].set_ylabel(latexstring("\\Delta s \\cdot t^{-2/3}"))
axs[2].set_xlabel(latexstring("t \\cdot U'^2"))
axs[2].set_ylabel(latexstring("z"))
plt.savefig("plots/su2_Uprime.$(format)", dpi=dpi)

# Sym-preserving Biquad. perturbations
perts = 0.4:0.4:2.0
plt.rc("axes", prop_cycle=plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))
fig,axs = plt.subplots(2,sharex=true)
for U in perts
    file = "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime0.0_U$(U)_mu0.001.h5"
    λ = U/(1+U)
    plot_hdf(axs, file, type="hdf", graph="both_transfer", label="J'=0.0, U=$(U)", t_scale=λ^6/(1-λ))
end
# axs[1].set_title("Magnetization transfer from initial domain wall (U perturbations around U=0.0)")
axs[1].set_xlabel(latexstring("t"))
axs[1].set_ylabel(latexstring("\\Delta s \\cdot t^{-2/3}"))
axs[2].set_xlabel(latexstring("t \\cdot \\frac{λ^6}{1-λ}"))
axs[2].set_ylabel(latexstring("z"))
plt.savefig("plots/su2_U.$(format)", dpi=dpi)

############ SU(4) point ##############

# Rung perturbations
perts = 0.1:0.1:0.5
plt.rc("axes", prop_cycle=plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))
fig,axs = plt.subplots(2,sharex=true)
for J2 in perts
    file = "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime$(J2)_U4.0_mu0.001.h5"
    plot_hdf(axs, file, type="hdf", graph="both_transfer", label="J'=$(J2), U=4.0")
end
# axs[1].set_title("Magnetization transfer from initial domain wall (J' perturbations around U=4.0)")
axs[1].set_xlabel(latexstring("t"))
axs[1].set_ylabel(latexstring("\\Delta s \\cdot t^{-2/3}"))
axs[2].set_xlabel(latexstring("t"))
axs[2].set_ylabel(latexstring("z"))
plt.savefig("plots/su4_J.$(format)", dpi=dpi)

# Sym-breaking Biquad. perturbations
perts = 0.4:0.4:1.6
plt.rc("axes", prop_cycle=plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))
fig,axs = plt.subplots(2,sharex=true)
for U in perts
    file = "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime0.0_U4.0_Uprime$(U)_mu0.001.h5"
    plot_hdf(axs, file, type="hdf", graph="both_transfer", label="J'=0.0, U=4.0, U'=$(U)", t_scale=U^2)
end
# axs[1].set_title("Magnetization transfer from initial domain wall (U' perturbations around U=4.0)")
axs[1].set_xlabel(latexstring("t"))
axs[1].set_ylabel(latexstring("\\Delta s \\cdot t^{-2/3}"))
axs[2].set_xlabel(latexstring("t \\cdot U'^2"))
axs[2].set_ylabel(latexstring("z"))
plt.savefig("plots/su4_Uprime.$(format)", dpi=dpi)

# Sym-preserving Biquad. perturbations
perts = [0.1,0.2,0.3,0.4,0.5,0.8,1.2,1.6,2.0]
plt.rc("axes", prop_cycle=plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))
fig,axs = plt.subplots(2,sharex=true)
for U in perts
    file = "data_plots/tdvp_coarsegrained_dw_gpu_L64_chi512_beta0.0_dt0.1_Jprime0.0_U$(U+4)_mu0.001.h5"
    plot_hdf(axs, file, type="hdf", graph="both_transfer", label="J'=0.0, U=$(U+4)", t_scale=U^2)
end
# axs[1].set_title("Magnetization transfer from initial domain wall (U perturbations around U=4.0)")
axs[1].set_xlabel(latexstring("t"))
axs[1].set_ylabel(latexstring("\\Delta s \\cdot t^{-2/3}"))
axs[2].set_xlabel(latexstring("t \\cdot δU^2"))
axs[2].set_ylabel(latexstring("z"))
plt.savefig("plots/su4_U.$(format)", dpi=dpi)

############ SU(3) ladder ##############

# Sym-preserving Biquad. perturbations
perts = [0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.4,1.6]
plt.rc("axes", prop_cycle=plt.cycler(color=[colormap(k) for k in LinRange(interval_low,interval_high,length(perts))]))
fig,axs = plt.subplots(2,sharex=true)
for U in perts
    file = "data_plots/tebd_su(3)_dw_L64_chi512_beta0.0_dt0.1_U$(U)_mu0.001_conserve_threesite_conj.h5"
    plot_hdf(axs, file, type="hdf", graph="both_transfer", label="U=$U", t_scale=U^6)
end
# axs[1].set_title("Magnetization transfer from initial domain wall (U perturbations)")
axs[1].set_xlabel(latexstring("t"))
axs[1].set_ylabel(latexstring("\\Delta s \\cdot t^{-2/3}"))
axs[2].set_xlabel(latexstring("t \\cdot U^6"))
axs[2].set_ylabel(latexstring("z"))
plt.savefig("plots/su3_U.$(format)", dpi=dpi)