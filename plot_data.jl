using CSV
using DataFrames
using PyPlot
using HDF5

# Plotting Maxime's data
function plot_csv(f::String)
    df=DataFrame(CSV.File(f, delim=" ", header=0, skipto=6))
    times = df.Column1
    corrs = 4 * abs.(df.Column2 + df.Column3*im)
    # corrs = 4 * df.Column3
    plt.loglog(times, corrs, label=f)
    plt.legend()
end

# Plot my own data
function plot_hdf(f::String, norm::Integer=1)
    F = h5open(f,"r")
    times = read(F, "times")
    corrs = norm * abs.(read(F, "corrs"))
    # ψ_norms = read(F, "psi_norms")
    # ψ2_norms = read(F, "psi2_norms")
    close(F)
    # corrs ./= ψ2_norms
    # corrs = norm * imag(read(F, "corrs"))
    plt.loglog(times, corrs, label=f)
    plt.legend()
end