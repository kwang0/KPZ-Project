using HDF5

# Copying data excluding large memory MPSs for easier transfer to local machine

# L = parse(Int64, ARGS[1])
# maxdim = parse(Int64, ARGS[2])
# β_max = parse(Float64, ARGS[3])
# δt = parse(Float64, ARGS[4])
# J2 = parse(Float64, ARGS[5])

L = 64
maxdim = 512
β_max = 0.0
δt = 0.1

for J2 in 0.1:0.1:0.5
    input = "data_jl/tdvp_L$(L)_chi$(maxdim)_beta$(β_max)_dt$(δt)_Jprime$(J2)_unnormed.h5"
    output = "data_plots/tdvp_L$(L)_chi$(maxdim)_beta$(β_max)_dt$(δt)_Jprime$(J2)_unnormed.h5"

    F = h5open(input,"r")
    times = read(F, "times")
    corrs = read(F, "corrs")
    ψ_norms = read(F, "psi_norms")
    ψ2_norms = read(F, "psi2_norms")

    G = h5open(output,"w")
    G["times"] = times
    G["corrs"] = corrs
    G["psi_norms"] = ψ_norms
    G["psi2_norms"] = ψ2_norms

    close(F)
    close(G)
end