using HDF5

# Copying data excluding large memory MPSs for easier transfer to local machine

# L = parse(Int64, ARGS[1])
# maxdim = parse(Int64, ARGS[2])
# β_max = parse(Float64, ARGS[3])
# δt = parse(Float64, ARGS[4])
# J2 = parse(Float64, ARGS[5])

L = 256
maxdim = 512
β_max = 0.0
δt = 0.1
J2 = 2.0

# for J2 in 0.1:0.1:0.5
    input = "/pscratch/sd/k/kwang98/KPZ/tebd_gpu_L256_chi1024_beta0.0_dt0.1_Jprime2.0_fullcorrs.h5"
    output = "data_plots/tebd_gpu_L256_chi1024_beta0.0_dt0.1_Jprime2.0_fullcorrs.h5"

    F = h5open(input,"r")
    times = read(F, "times")
    corrs = read(F, "corrs")
    ψ_norms = read(F, "psi_norms")
    ψ2_norms = read(F, "psi2_norms")
    close(F)

    G = h5open(output,"w")
    G["times"] = times
    G["corrs"] = corrs
    G["psi_norms"] = ψ_norms
    G["psi2_norms"] = ψ2_norms
    close(G)
# end