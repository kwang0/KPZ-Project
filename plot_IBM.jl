using HDF5
using PyPlot

fig, axs = plt.subplots(2, figsize=(7,10))
ax = axs[1]
for J2 in [0.0,1.0,2.0,4.0]
    f = "data_plots/IBM_sims_L32_chi512_dt1.0_offset5_rungX0.0_rungY0.0_rungZ$(J2).h5"
    F = h5open(f,"r")
    times = read(F, "times")
    corrs = real(read(F, "corrs"))
    close(F)
    # ax.loglog(times[1:19], 0.5*real.(corrs[1,1:19]), label=latexstring("\$\\frac{J_\\perp}{J} = $J2\$"))
    ax.loglog(times[1:19], 0.5*real.(sum(corrs, dims=1)[1:19]), label=latexstring("\$\\frac{J_\\perp}{J} = $J2\$"))
end
ax.set_xlabel("t")
# ax.set_ylabel(latexstring("\$C_{00}(t)\$"))
ax.set_ylabel(latexstring("\$\\sum_iC_{i0}(t)\$"))
ax.set_ylim(1e-2,2)
ax.set_title(latexstring("\$ZZ\$ interactions"))

ax = axs[2]
for J2 in [0.0,1.0,2.0,4.0]
    f = "data_plots/IBM_sims_L32_chi512_dt1.0_offset5_rungX$(J2)_rungY$(J2)_rungZ0.0.h5"
    F = h5open(f,"r")
    times = read(F, "times")
    corrs = real(read(F, "corrs"))
    close(F)
    # ax.loglog(times[1:19], 0.5*real.(corrs[1,1:19]), label=latexstring("\$\\frac{J_\\perp}{J} = $J2\$"))
    ax.loglog(times[1:19], 0.5*real.(sum(corrs, dims=1)[1:19]), label=latexstring("\$\\frac{J_\\perp}{J} = $J2\$"))
end
ax.legend()
ax.set_xlabel("t")
# ax.set_ylabel(latexstring("\$C_{00}(t)\$"))
ax.set_ylabel(latexstring("\$\\sum_iC_{i0}(t)\$"))
ax.set_ylim(1e-2,2)
ax.set_title(latexstring("\$XX+YY\$ interactions"))
fig.tight_layout()

fig, axs = plt.subplots(2, figsize=(7,10))
ax = axs[1]
for J2 in [0.0,1.0,2.0,4.0]
    f = "data_plots/IBM_sims_L32_chi512_dt1.0_offset5_rungX$(J2)_rungY0.0_rungZ0.0.h5"
    F = h5open(f,"r")
    times = read(F, "times")
    corrs = real(read(F, "corrs"))
    close(F)
    # ax.loglog(times[1:19], 0.5*real.(corrs[1,1:19]), label=latexstring("\$\\frac{J_\\perp}{J} = $J2\$"))
    ax.loglog(times[1:19], 0.5*real.(sum(corrs, dims=1)[1:19]), label=latexstring("\$\\frac{J_\\perp}{J} = $J2\$"))
end
ax.set_xlabel("t")
# ax.set_ylabel(latexstring("\$C_{00}(t)\$"))
ax.set_ylabel(latexstring("\$\\sum_iC_{i0}(t)\$"))
ax.set_ylim(1e-2,2)
ax.set_title(latexstring("\$XX\$ interactions"))

ax = axs[2]
for J2 in [0.0,1.0,2.0,4.0]
    f = "data_plots/IBM_sims_L32_chi512_dt1.0_offset5_rungX$(J2)_rungY0.0_rungZ$(J2).h5"
    F = h5open(f,"r")
    times = read(F, "times")
    corrs = real(read(F, "corrs"))
    close(F)
    # ax.loglog(times[1:19], 0.5*real.(corrs[1,1:19]), label=latexstring("\$\\frac{J_\\perp}{J} = $J2\$"))
    ax.loglog(times[1:19], 0.5*real.(sum(corrs, dims=1)[1:19]), label=latexstring("\$\\frac{J_\\perp}{J} = $J2\$"))
end
ax.legend()
ax.set_xlabel("t")
# ax.set_ylabel(latexstring("\$C_{00}(t)\$"))
ax.set_ylabel(latexstring("\$\\sum_iC_{i0}(t)\$"))
ax.set_ylim(1e-2,2)
ax.set_title(latexstring("\$XX+ZZ\$ interactions"))
fig.tight_layout()