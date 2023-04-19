using MKL
using LinearAlgebra
using ITensors
using Printf
using PyPlot
using HDF5

function ITensors.op(::OpName"expiSS", ::SiteType"S=1/2", s1::Index, s2::Index; t)
  h =
    1 / 2 * op("S+", s1) * op("S-", s2) +
    1 / 2 * op("S-", s1) * op("S+", s2) +
    op("Sz", s1) * op("Sz", s2)
  return exp(-im * t * h)
end

function fourth_order_trotter_gates(L, sites, δt)
  α = 1 / (4 - 4^(1/3))

  A1 = ops([("expiSS", (n, n + 1), (t=α*δt/2,)) for n in 1:2:(L - 1)], sites)
  B1 = ops([("expiSS", (n, n + 1), (t=α*δt,)) for n in 2:2:(L - 2)], sites)
  A2 = ops([("expiSS", (n, n + 1), (t=α*δt,)) for n in 1:2:(L - 1)], sites)
  A3 = ops([("expiSS", (n, n + 1), (t=(1-3*α)*δt/2,)) for n in 1:2:(L - 1)], sites)
  B2 = ops([("expiSS", (n, n + 1), (t=(1-4*α)*δt,)) for n in 2:2:(L - 2)], sites)

  return vcat(A1,B1,A2,B1,A3,B2,A3,B1,A2,B1,A1)
end

function main(; L=128, cutoff=1E-16, δτ=0.05, beta_max=3.0, δt=0.1, ttotal=100, maxdim=32)
  s = siteinds("S=1/2", L; conserve_qns=true)

  # Make purification gates
  im_gates = fourth_order_trotter_gates(L, s, -im * δτ)

  # Initial state is infinite-temperature mixed state
  rho = MPO(s, "Id") ./ √2

  # Do the time evolution by applying the gates
  # for Nsteps steps
  for β in 0:δτ:beta_max/2
    @printf("β = %.2f\n", 2*β)
    flush(stdout)
    rho = apply(im_gates, rho; cutoff, maxdim)
    normalize!(rho)
  end

  # Make real-time evolution gates
  real_gates = fourth_order_trotter_gates(L, s, δt)

  c = div(L, 2) # center site
  Sz_center = op("Sz",s[c])
  rho2 = apply(Sz_center, rho; cutoff, maxdim)
  normalize!(rho2)

  times = Float64[]
  corrs = ComplexF64[]
  for t in 0.0:δt:ttotal
    rho3 = apply(Sz_center, rho2; cutoff, maxdim)
    normalize!(rho3)
    corr = inner(rho, rho3)
    println("$t $corr")
    flush(stdout)
    push!(times, t)
    push!(corrs, corr)

    # Writing to data file
    F = h5open("data_jl/tebd_L$(L)_chi$(maxdim)_beta$(beta_max)_dt$(δt)_order4.h5","w")
    F["times"] = times
    F["corrs"] = corrs
    close(F)

    t≈ttotal && break

    rho = apply(real_gates, rho; cutoff, maxdim)
    normalize!(rho)
    rho2 = apply(real_gates, rho2; cutoff, maxdim)
    normalize!(rho2)
  end

  # plt.loglog(times, abs.(corrs))
  # plt.xlabel("t")
  # plt.ylabel("|C(T,x=0,t)|")
  # plt.show()

  return times, corrs
end

ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(40)
# ITensors.enable_threaded_blocksparse(true)

L = parse(Int64, ARGS[1])
maxdim = parse(Int64, ARGS[2])
beta_max = parse(Float64, ARGS[3])
δt = parse(Float64, ARGS[4])

main(L=L, maxdim=maxdim, beta_max=beta_max, δt=δt)