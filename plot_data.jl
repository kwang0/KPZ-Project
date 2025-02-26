using CSV
using DataFrames
using PyPlot
using HDF5
using CurveFit
using Glob

# # Plotting Maxime's data
# function plot_csv(f::String)
#     df=DataFrame(CSV.File(f, delim=" ", header=0, skipto=6))
#     times = df.Column1
#     corrs = 4 * abs.(df.Column2 + df.Column3*im)
#     # corrs = 4 * df.Column3
#     plt.loglog(times, corrs .* (times .^ (2/3)) , label=f)
#     plt.legend()
# end

# Plot my own data
function plot_hdf(ax, f::String; norm::Float64=1.0, type = "hdf", graph="twosite", t_scale=1.0, label="default", dw="Z", ncol=1)
    if label == "default"
        label = f
    end
    if graph == "both"
        plot_hdf(ax[1], f, type=type, graph = "twosite", label = label)
        plot_hdf(ax[2], f, type=type, graph = "exponent", label = label)
        return
    elseif graph == "both_transfer"
        plot_hdf(ax[1], f, type=type, graph = "transfer", t_scale=1.0, label = label, dw = dw, ncol=ncol)
        plot_hdf(ax[2], f, type=type, graph = "exponent_transfer", t_scale=t_scale, label = label, dw = dw, ncol=ncol)
        return
    end

    if type == "hdf"
        F = h5open(f,"r")
        times = read(F, "times")
        corrs = norm * real(read(F, "corrs"))
        # corrs ./= abs.(sum(corrs, dims=1))
        close(F)
    elseif type == "csv"
        df=DataFrame(CSV.File(f, delim=" ", header=0, skipto=6))
        times = df.Column1
        corrs = 4 * abs.(df.Column2 + df.Column3*im)
    end
        
    # if length(size(corrs)) == 2
    if graph == "onesite"
        type == "hdf" && (length(size(corrs)) == 2) && (corrs = corrs[size(corrs)[1]÷2-1, :])
        ax.loglog(times, corrs , label=f)
    elseif graph == "twosite"
        type == "hdf" && (length(size(corrs)) == 2) && (corrs = corrs[size(corrs)[1]÷2-1, :] + corrs[size(corrs)[1]÷2, :])

        # sigma = 2.0
        # broadening = (1.0 / (sigma * sqrt(2 * pi))) .* exp.(-(times .- times').^2 ./ (2*sigma^2))
        # corrs = broadening * corrs

        ax.loglog(times, corrs .* (times.^(2/3)), label=f)
        # ax.legend()
    elseif graph == "exponent"
        type == "hdf" && (length(size(corrs)) == 2) && (corrs = corrs[size(corrs)[1]÷2-1, :] + corrs[size(corrs)[1]÷2, :])
        # alphas = (log.(corrs[2:end]) .- log.(corrs[1:end-1])) ./ (log.(times[2:end]) .- log.(times[1:end-1]))
        # plt.plot(times[1:end-1], alphas, label=f)

        # sigma = 2.0
        # broadening = (1.0 / (sigma * sqrt(2 * pi))) .* exp.(-(times .- times').^2 ./ (2*sigma^2))
        # corrs = broadening * corrs

        alphas = []
        errors = []
        ts = []
        t = 2.5
        scale = 1.25
        while (t < times[end] && size(times[times .> t],1) > 100)
            push!(ts, t)
            window_min = t
            window_max = t + 20

            window_corrs = corrs[(times .> window_min) .& (times .< window_max)]
            window_times = times[(times .> window_min) .& (times .< window_max)]
            x = log.(window_times)
            y = log.(window_corrs)

            b, m = linear_fit(x, y)
            m_err = sqrt(sum((m .* x .+ b .- y).^2) / (sum((x .- (sum(x) / size(x,1))).^2) * (size(x,1)-2)))
            alpha = -1.0/m
            alpha_err = m_err/m^2

            push!(alphas, alpha)
            push!(errors, alpha_err)
            t *= scale
        end
        # ax.scatter(ts, alphas, label=f, s=10.0, marker="x")
        ax.set_ylim(1,2)
        ax.errorbar(ts, alphas, yerr=errors, label=f, marker=".")
        ax.legend()
    elseif graph == "full"
        # fig, ax = plt.subplots(1)
        L = size(corrs,1)
        img = matplotlib[:image][:NonUniformImage](ax, interpolation="nearest", cmap="hot", extent=(1,L,0,times[end]))
        img.set_data(times.^(2/3), LinRange(1,L,L), log.(corrs))
        ax.add_image(img)
        ax.set_xlim(0,times[end])
        ax.set_ylim(1,L)
        plt.title(f)
    elseif graph == "fulldw"
        F = h5open(f,"r")
        Z1s = real(read(F, "Z1s"))
        Z2s = real(read(F, "Z2s"))
        close(F)

        Zs = (Z1s .+ Z2s)

        # times .= times.^(2/3)
        L = size(Zs,1)
        img = matplotlib[:image][:NonUniformImage](ax, interpolation="nearest", cmap="hot", extent=(1,L,0,times[end]))
        img.set_data(times, LinRange(1,L,L), Zs)
        ax.add_image(img)
        ax.set_xlim(0,times[end])
        ax.set_ylim(1,L)
        fig.colorbar(img, ax=ax)
        plt.title(f)
    elseif graph == "drude"
        corrs = sum(corrs, dims=1)[:]
        total = 0.0
        diffusion = []
        for corr in corrs
            total += corr
            push!(diffusion, total)
        end
        ax.plot(times .^ (1/3), diffusion, label=f)
        # plt.plot(log.(times), diffusion, label=f)
        ax.legend()
    elseif graph == "transfer"
        if dw == "Z"
            F = h5open(f,"r")
            Z1s = real(read(F, "Z1s"))
            Z2s = real(read(F, "Z2s"))
            close(F)
            Qs = (Z1s .+ Z2s)
        elseif dw == "rung"
            F = h5open(f,"r")
            Qs = real(read(F, "Qs"))
            close(F)
        elseif dw == "su(3)"
            F = h5open(f,"r")
            Qs = real(read(F, "Zs"))
            close(F)
        end

        c = size(Qs,1)÷2
        transfer = sum(Qs[1:c,:],dims=1)
        transfer .= transfer[1] .- transfer
        ax.loglog(times .* t_scale, transfer[:] .* (times .^ (-2/3)), label = label)
        # plt.plot(log.(times), diffusion, label=f)
        ax.legend(ncol=ncol)
    elseif graph == "exponent_transfer"
        if dw == "Z"
            F = h5open(f,"r")
            Z1s = real(read(F, "Z1s"))
            Z2s = real(read(F, "Z2s"))
            close(F)
            Qs = (Z1s .+ Z2s)
        elseif dw == "rung"
            F = h5open(f,"r")
            Qs = real(read(F, "Qs"))
            close(F)
        elseif dw == "su(3)"
            F = h5open(f,"r")
            Qs = real(read(F, "Zs"))
            close(F)
        end

        c = size(Qs,1)÷2
        transfer = sum(Qs[1:c,:],dims=1)
        transfer .= transfer[1] .- transfer

        alphas = []
        errors = []
        ts = []
        t = 2.5
        scale = 1.25
        while (t < times[end] && size(times[times .> t],1) > 100)
            push!(ts, t)
            window_min = t
            window_max = t + 20

            window_transfer = transfer[(times .> window_min) .& (times .< window_max)]
            window_times = times[(times .> window_min) .& (times .< window_max)] .* t_scale
            x = log.(window_times)
            y = log.(window_transfer)

            b, m = linear_fit(x, y)
            m_err = sqrt(sum((m .* x .+ b .- y).^2) / (sum((x .- (sum(x) / size(x,1))).^2) * (size(x,1)-2)))
            alpha = 1.0/m
            alpha_err = m_err/m^2

            push!(alphas, alpha)
            push!(errors, alpha_err)
            t *= scale
        end
        ax.set_xscale("log")
        ax.set_yticks([1,1.5,2],["1","1.5","2"])
        ax.plot(ts .* t_scale, alphas, label=f, marker=".", linestyle="--")
        ax.set_ylim(1,2)
        # ax.errorbar(ts .* t_scale, alphas, yerr=errors, label=f, marker=".", linestyle="--")
        # ax.legend()
    elseif graph == "entropy"
        F = h5open(f,"r")
        Ss = real(read(F, "Ss"))
        close(F)
        
        ax.plot(times, Ss[:], label=f)
    end
    # end
    
end

function plot_fit(f::String, window_min, window_max; graph="twosite")
    F = h5open(f,"r")
    times = read(F, "times")
    corrs = abs.(read(F, "corrs"))
    if length(size(corrs)) == 2
        if graph == "onesite"
            corrs = corrs[size(corrs)[1]÷2-1, :]
        elseif graph == "twosite"
            corrs = corrs[size(corrs)[1]÷2-1, :] + corrs[size(corrs)[1]÷2, :]
        elseif graph == "drude"
            corrs = sum(corrs, dims=1)[:]
        end
    end
    close(F)
    corrs = corrs[(times .> window_min) .& (times .< window_max)]
    times = times[(times .> window_min) .& (times .< window_max)]

    b, m = linear_fit(log.(times), log.(corrs))

    plt.loglog(times, exp.(m .* (log.(times)) .+ b), label="z = $(-1.0/m)")
    plt.legend()
end

function plot_fit(f::String)
    t = 10.0
    F = h5open(f,"r")
    times = read(F, "times")
    close(F)

    scale = 2
    while (t < times[end])
        plot_fit(f, t, scale * t)
        t *= scale
    end
end
