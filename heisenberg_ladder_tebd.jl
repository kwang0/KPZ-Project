using MKL
using LinearAlgebra
using ITensors
using ITensorTDVP
using Printf
using PyPlot
using HDF5

function inf_temp_mps(sites)
  num_sites = length(sites)
  if (num_sites % 2 != 0)
    throw(DomainError(num_sites,"Expects even number of sites for ancilla-physical singlets."))
  else
    state = [isodd(n) ? "Up" : "Dn" for n=1:num_sites]
    ψ = MPS(sites, state) # Initialize as Neel state to get correct QNs between singlets
    for j = 1:2:num_sites-1
      s1 = sites[j]
      s2 = sites[j+1]
          
      if(j == 1)
        rightlink = commonind(ψ[j+1],ψ[j+2])
        A = ITensor(ComplexF64, s1, s2, rightlink)

        A[s1=>1, s2=>2, rightlink => 1] = 1/sqrt(2)
        A[s1=>2, s2=>1, rightlink => 1] = -1/sqrt(2)

        U,S,V = svd(A, (s1), cutoff=1e-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V

      elseif (j == num_sites-1)
        leftlink = dag(commonind(ψ[j-1], ψ[j]))
        A = ITensor(ComplexF64, s1, s2, leftlink)

        A[s1 => 1,s2 => 2, leftlink => 1] = 1/sqrt(2)
        A[s1 => 2,s2 => 1, leftlink => 1] = -1/sqrt(2)

        U,S,V = svd(A, (s1, leftlink), cutoff=1e-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V
        
      else
        rightlink = commonind(ψ[j+1], ψ[j+2])
        leftlink = dag(commonind(ψ[j-1], ψ[j]))
    
        A = ITensor(ComplexF64, s1, s2, rightlink, leftlink)

        A[s1 => 1,s2 => 2, rightlink=>1, leftlink =>1] = 1/sqrt(2)
        A[s1 => 2,s2 => 1, rightlink=>1, leftlink =>1] = -1/sqrt(2)

        U,S,V = svd(A, (s1, leftlink), cutoff=1e-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V
      end
    end

    return ψ
  end
end

function ITensors.op(::OpName"expiSS", ::SiteType"S=1/2", s1::Index, s2::Index; t)
  h =
    1 / 2 * op("S+", s1) * op("S-", s2) +
    1 / 2 * op("S-", s1) * op("S+", s2) +
    op("Sz", s1) * op("Sz", s2)
  return cuITensor(exp(-im * t * h))
end

# function fourth_order_trotter_gates(L, sites, δt, real_evolution)
#   α = 1 / (4 - 4^(1/3))

#   A1 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=α*δt/2,)) for n in 1:2:(L - 1)], sites)
#   B1 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=α*δt,)) for n in 2:2:(L - 2)], sites)
#   A2 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=α*δt,)) for n in 1:2:(L - 1)], sites)
#   A3 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=(1-3*α)*δt/2,)) for n in 1:2:(L - 1)], sites)
#   B2 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=(1-4*α)*δt,)) for n in 2:2:(L - 2)], sites)

#   if (real_evolution)
#     # Apply disentangler exp(iHt) on ancilla sites
#     aA1 = ops([("expiSS", (2*n, 2*n + 2), (t=-α*δt/2,)) for n in 1:2:(L - 1)], sites)
#     aB1 = ops([("expiSS", (2*n, 2*n + 2), (t=-α*δt,)) for n in 2:2:(L - 2)], sites)
#     aA2 = ops([("expiSS", (2*n, 2*n + 2), (t=-α*δt,)) for n in 1:2:(L - 1)], sites)
#     aA3 = ops([("expiSS", (2*n, 2*n + 2), (t=-(1-3*α)*δt/2,)) for n in 1:2:(L - 1)], sites)
#     aB2 = ops([("expiSS", (2*n, 2*n + 2), (t=-(1-4*α)*δt,)) for n in 2:2:(L - 2)], sites)

#     A1 = vcat(A1,aA1)
#     B1 = vcat(B1,aB1)
#     A2 = vcat(A2,aA2)
#     A3 = vcat(A3,aA3)
#     B2 = vcat(B2,aB2)
#   end

#   return vcat(A1,B1,A2,B1,A3,B2,A3,B1,A2,B1,A1)
# end

function fourth_order_trotter_gates(L, sites, δt, real_evolution)
  a1 = 0.095848502741203681182
  a2 = -0.078111158921637922695
  a3 = 0.5 - (a1 + a2)
  b1 = 0.42652466131587616168
  b2 = -0.12039526945509726545
  b3 = 1 - 2 * (b1 + b2)

  A1 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=a1*δt,)) for n in 1:2:(L - 1)], sites)
  B1 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=b1*δt,)) for n in 2:2:(L - 2)], sites)
  A2 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=a2*δt,)) for n in 1:2:(L - 1)], sites)
  B2 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=b2*δt,)) for n in 2:2:(L - 2)], sites)
  A3 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=a3*δt,)) for n in 1:2:(L - 1)], sites)
  B3 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=b3*δt,)) for n in 2:2:(L - 2)], sites)

  if (real_evolution)
    # Apply disentangler exp(iHt) on ancilla sites
    aA1 = ops([("expiSS", (2*n, 2*n + 2), (t=-a1*δt,)) for n in 1:2:(L - 1)], sites)
    aB1 = ops([("expiSS", (2*n, 2*n + 2), (t=-b1*δt,)) for n in 2:2:(L - 2)], sites)
    aA2 = ops([("expiSS", (2*n, 2*n + 2), (t=-a2*δt,)) for n in 1:2:(L - 1)], sites)
    aB2 = ops([("expiSS", (2*n, 2*n + 2), (t=-b2*δt,)) for n in 2:2:(L - 2)], sites)
    aA3 = ops([("expiSS", (2*n, 2*n + 2), (t=-a3*δt,)) for n in 1:2:(L - 1)], sites)
    aB3 = ops([("expiSS", (2*n, 2*n + 2), (t=-b3*δt,)) for n in 2:2:(L - 2)], sites)

    A1 = vcat(A1,aA1)
    B1 = vcat(B1,aB1)
    A2 = vcat(A2,aA2)
    B2 = vcat(B2,aB2)
    A3 = vcat(A3,aA3)
    B3 = vcat(B3,aB3)
  end

  return vcat(A1,B1,A2,B2,A3,B3,A3,B2,A2,B1,A1)
end

function main(; L=128, cutoff=1E-16, δτ=0.05, beta_max=3.0, δt=0.1, ttotal=100, maxdim=32)
  s = siteinds("S=1/2", 2 * L; conserve_qns=false)

  # Make purification gates
  im_gates = fourth_order_trotter_gates(L, s, -im * δτ, false)

  # Initial state is infinite-temperature mixed state (purification)
  ψ = cuMPS(inf_temp_mps(s))

  # Cool down to inverse temperature 
  for β in δτ:δτ:beta_max/2
    @printf("β = %.2f\n", 2*β)
    flush(stdout)
    ψ = apply(im_gates, ψ; cutoff, maxdim)
    normalize!(ψ)
  end

  # Make real-time evolution gates
  real_gates = fourth_order_trotter_gates(L, s, δt, true)

  c = div(L, 2) # center site
  Sz_center = cuITensor(op("Sz",s[2*c-1]))
  ψ2 = apply(2 * Sz_center, ψ; cutoff, maxdim)
  # normalize!(ψ2)

  times = Float64[]
  corrs = ComplexF64[]
  for t in 0.0:δt:ttotal
    ψ3 = apply(2 * Sz_center, ψ2; cutoff, maxdim)
    # normalize!(ψ3)
    corr = inner(ψ, ψ3)
    println("$t $corr")
    flush(stdout)
    push!(times, t)
    push!(corrs, corr)

    # Writing to data file
    F = h5open("data_jl/tebd_L$(L)_chi$(maxdim)_beta$(beta_max)_dt$(δt)_apply_unnormed.h5","w")
    F["times"] = times
    F["corrs"] = corrs
    close(F)

    t≈ttotal && break

    ψ = apply(real_gates, ψ; cutoff, maxdim)
    # normalize!(ψ)
    ψ2 = apply(real_gates, ψ2; cutoff, maxdim)
    # normalize!(ψ2)
  end

  # plt.loglog(times, abs.(corrs))
  # plt.xlabel("t")
  # plt.ylabel("|C(T,x=0,t)|")
  # plt.show()

  return times, corrs
end

# Set to identity to run on CPU
gpu = cu

ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(1)
# ITensors.enable_threaded_blocksparse(true)

L = parse(Int64, ARGS[1])
maxdim = parse(Int64, ARGS[2])
beta_max = parse(Float64, ARGS[3])
δt = parse(Float64, ARGS[4])

main(L=L, maxdim=maxdim, beta_max=beta_max, δt=δt)