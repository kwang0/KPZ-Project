using HDF5

# Copying data excluding large memory MPSs for easier transfer to local machine

# L = parse(Int64, ARGS[1])
# maxdim = parse(Int64, ARGS[2])
# β_max = parse(Float64, ARGS[3])
# δt = parse(Float64, ARGS[4])
# J2 = parse(Float64, ARGS[5])

L = 128
maxdim = 1024
β_max = 0.0
δt = 0.5
J2 = 2.0

file = "tdvp_L$(L)_chi$(maxdim)_beta$(β_max)_dt$(δt)_Jprime$(J2).h5"
input = "/pscratch/sd/k/kwang98/KPZ/" * file
output = "data_plots/" * file

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