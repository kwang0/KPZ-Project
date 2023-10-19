using CSV
using DataFrames
using PyPlot
using HDF5
using CurveFit

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
function plot_hdf(f::String, norm::Float64=1.0)
    F = h5open(f,"r")
    times = read(F, "times")
    corrs = norm * abs.(read(F, "corrs"))
    # corrs ./= abs.(sum(corrs, dims=1))
    if length(size(corrs)) == 2
        corrs = corrs[size(corrs)[1]÷2-1, :]
    end
    # ψ_norms = read(F, "psi_norms")
    # ψ2_norms = read(F, "psi2_norms")
    close(F)
    # corrs ./= ψ2_norms
    # corrs = norm * imag(read(F, "corrs"))
    plt.loglog(times, corrs, label=f)
    plt.legend()
end

function plot_fit(f::String, window_min, window_max)
    F = h5open(f,"r")
    times = read(F, "times")
    corrs = abs.(read(F, "corrs"))
    if length(size(corrs)) == 2
        corrs = corrs[size(corrs)[1]÷2-1, :]
    end
    close(F)
    corrs = corrs[(times .> window_min) .& (times .< window_max)]
    times = times[(times .> window_min) .& (times .< window_max)]

    b, m = linear_fit(log.(times), log.(corrs))

    plt.loglog(times, exp.(m .* (log.(times)) .+ b), label="z = $(-1.0/m)")
    plt.legend()
end