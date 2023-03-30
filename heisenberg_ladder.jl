using ITensors
using Printf
using PyPlot
using HDF5

function ITensors.op(::OpName"expτSS", ::SiteType"S=1/2", s1::Index, s2::Index; τ)
  h =
    1 / 2 * op("S+", s1) * op("S-", s2) +
    1 / 2 * op("S-", s1) * op("S+", s2) +
    op("Sz", s1) * op("Sz", s2)
  return exp(τ * h)
end

function ITensors.op(::OpName"expiSS", ::SiteType"S=1/2", s1::Index, s2::Index; t)
    h =
      1 / 2 * op("S+", s1) * op("S-", s2) +
      1 / 2 * op("S-", s1) * op("S+", s2) +
      op("Sz", s1) * op("Sz", s2)
    return exp(-im * t * h)
  end

function main(; L=128, cutoff=1E-8, δτ=0.05, beta_max=3.0, δt=0.1, ttotal=100, maxdim=32)
  s = siteinds("S=1/2", L; conserve_qns=true)

  # Make purification gates (1,2),(2,3),(3,4),...
  im_gates = ops([("expτSS", (n, n + 1), (τ=-δτ / 2,)) for n in 1:(L - 1)], s)
  # Include gates in reverse order to complete Trotter formula
  append!(im_gates, reverse(im_gates))

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

  # Make real gates
  real_gates = ops([("expiSS", (n, n + 1), (t=δt / 2,)) for n in 1:(L - 1)], s)
  # Include gates in reverse order to complete Trotter formula
  append!(real_gates, reverse(real_gates))

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
    F = h5open("data_jl/mpo_L$(L)_chi$(maxdim)_beta$(beta_max)_dt$δt.h5","w")
    F["times"] = times
    F["corrs"] = corrs
    close(F)

    t≈ttotal && break

    rho = apply(real_gates, rho; cutoff, maxdim)
    normalize!(rho)
    rho2 = apply(real_gates, rho2; cutoff, maxdim)
    normalize!(rho2)
  end

  plt.loglog(times, abs.(corrs))
  plt.xlabel("t")
  plt.ylabel("|C(T,x=0,t)|")
  plt.show()

  return times, corrs
end

L = parse(Int64, ARGS[1])
maxdim = parse(Int64, ARGS[2])
beta_max = parse(Float64, ARGS[3])
δt = parse(Float64, ARGS[4])

main(L=L, maxdim=maxdim, beta_max=beta_max, δt=δt)