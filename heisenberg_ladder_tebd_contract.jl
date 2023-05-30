using MKL
using LinearAlgebra
using ITensors
using ITensorGPU
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
  return exp(-im * t * h)
end

function create_gates(L, sites, t)
  gates = ITensor[]
  for i in 1:2:(2*L-3)
    # push!(gates, op("expiSS", sites[i], sites[i+2], t=t))
    # push!(gates, op("expiSS", sites[i+1], sites[i+3], t=-t))
    push!(gates, cuITensor(op("expiSS", sites[i], sites[i+2], t=t)))
    push!(gates, cuITensor(op("expiSS", sites[i+1], sites[i+3], t=-t)))
  end

  return gates
end

function create_gate_list(L, sites, t)
  a1 = 0.095848502741203681182
  a2 = -0.078111158921637922695
  a3 = 0.5 - (a1 + a2)
  b1 = 0.42652466131587616168
  b2 = -0.12039526945509726545
  b3 = 1 - 2 * (b1 + b2)

  gate_list = Vector{Vector{ITensor}}(undef,6)

  gate_list[1] = create_gates(L, sites, a1 * t)
  gate_list[2] = create_gates(L, sites, b1 * t)
  gate_list[3] = create_gates(L, sites, a2 * t)
  gate_list[4] = create_gates(L, sites, b2 * t)
  gate_list[5] = create_gates(L, sites, a3 * t)
  gate_list[6] = create_gates(L, sites, b3 * t)

  return gate_list
end

function trotter_sweep(ψ, L, sites, start, gates, cut, m, alg)
  # Initial edge case
  if (start == 5)
    orthogonalize!(ψ, 1; cutoff=cut, maxdim=m)
    three_site = ψ[1]*ψ[2]*ψ[3]
    three_site = apply(gates[1], three_site)
    U,S,V = svd(three_site, (sites[1]), cutoff=cut, maxdim=m, alg=alg, lefttags="Link,l=1")
    ψ[1] = U

    three_site = S*V * ψ[4]
    three_site = apply(gates[2], three_site)
    leftlink = dag(commonind(ψ[1], three_site))
    U,S,V = svd(three_site, (sites[2], leftlink), cutoff=cut, maxdim=m, alg=alg, lefttags="Link,l=2")
    ψ[2] = U

    two_site = S*V
    leftlink = dag(commonind(ψ[2], two_site))
    U,S,V = svd(two_site, (sites[3], leftlink), cutoff=cut, maxdim=m, alg=alg, lefttags="Link,l=3")
    ψ[3] = U
    ψ[4] = S*V
  end

  last = start + 2*L - 8

  for i in start:4:last
    orthogonalize!(ψ, i; cutoff=cut, maxdim=m)
    three_site = ψ[i] * ψ[i+1] * ψ[i+2]
    three_site = apply(gates[i], three_site)
    leftlink = dag(commonind(ψ[i-1], three_site))
    U,S,V = svd(three_site, (sites[i], leftlink), cutoff=cut, maxdim=m, alg=alg, lefttags="Link,l=$(i)")
    ψ[i] = U

    three_site = S*V * ψ[i+3]
    three_site = apply(gates[i+1], three_site)
    leftlink = dag(commonind(ψ[i], three_site))
    U,S,V = svd(three_site, (sites[i+1], leftlink), cutoff=cut, maxdim=m, alg=alg, lefttags="Link,l=$(i+1)")
    ψ[i+1] = U

    two_site = S*V
    leftlink = dag(commonind(ψ[i+1], two_site))
    U,S,V = svd(two_site, (sites[i+2], leftlink), cutoff=cut, maxdim=m, alg=alg, lefttags="Link,l=$(i+2)")
    ψ[i+2] = U
    ψ[i+3] = S*V
  end

  return ψ
end

function fourth_order_contract(ψ, L, sites, gate_list, cut, m)
  alg = "qr_iteration"

  ψ = trotter_sweep(ψ, L, sites, 5, gate_list[1], cut, m, alg)
  ψ = trotter_sweep(ψ, L, sites, 3, gate_list[2], cut, m, alg)
  ψ = trotter_sweep(ψ, L, sites, 5, gate_list[3], cut, m, alg)
  ψ = trotter_sweep(ψ, L, sites, 3, gate_list[4], cut, m, alg)
  ψ = trotter_sweep(ψ, L, sites, 5, gate_list[5], cut, m, alg)
  ψ = trotter_sweep(ψ, L, sites, 3, gate_list[6], cut, m, alg)
  ψ = trotter_sweep(ψ, L, sites, 5, gate_list[5], cut, m, alg)
  ψ = trotter_sweep(ψ, L, sites, 3, gate_list[4], cut, m, alg)
  ψ = trotter_sweep(ψ, L, sites, 5, gate_list[3], cut, m, alg)
  ψ = trotter_sweep(ψ, L, sites, 3, gate_list[2], cut, m, alg)
  ψ = trotter_sweep(ψ, L, sites, 5, gate_list[1], cut, m, alg)
  
  return ψ
end

function main(; L=128, cutoff=1E-12, δτ=0.05, beta_max=3.0, δt=0.1, ttotal=100, maxdim=32)
  sites = siteinds("S=1/2", 2 * L; conserve_qns=false)

  # Initial state is infinite-temperature mixed state (purification)
  # ψ = inf_temp_mps(sites)
  ψ = cuMPS(inf_temp_mps(sites))

  # Cool down to inverse temperature 
  for β in δτ:δτ:beta_max/2
    @printf("β = %.2f\n", 2*β)
    flush(stdout)
    ψ = trotter_sweep(ψ, L, sites, (1/2) * δτ, false, cutoff, maxdim)
    # ψ = trotter_sweep(ψ, L, s, ((1-2*α)/2) * δτ, false, cutoff, maxdim)
    # ψ = trotter_sweep(ψ, L, s, (α/2) * δτ, false, cutoff, maxdim)
    normalize!(ψ)
  end

  c = div(L, 2) # center site
  # Sz_center = op("Sz",sites[2*c-1])
  Sz_center = cuITensor(op("Sz",sites[2*c-1]))
  ψ2 = apply(2 * Sz_center, ψ; cutoff, maxdim)
  # normalize!(ψ2)

  real_gate_list = create_gate_list(L, sites, δt)

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
    F = h5open("data_jl/tebd_L$(L)_chi$(maxdim)_beta$(beta_max)_dt$(δt)_gpu.h5","w")
    F["times"] = times
    F["corrs"] = corrs
    close(F)

    t≈ttotal && break

    ψ = @time fourth_order_contract(ψ, L, sites, real_gate_list, cutoff, maxdim)
    # normalize!(ψ)
    ψ2 = @time fourth_order_contract(ψ2, L, sites, real_gate_list, cutoff, maxdim)
    # normalize!(ψ2)

    println("Max bond dimension is $(maxlinkdim(ψ2))")
    println("Norm of psi2 is $(norm(ψ2))")
  end

  plt.loglog(times, abs.(corrs))
  plt.xlabel("t")
  plt.ylabel("|C(T,x=0,t)|")
  plt.show()

  return times, corrs
end

# Set to identity to run on CPU
gpu = cu

ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(40)
# ITensors.enable_threaded_blocksparse(true)

L = parse(Int64, ARGS[1])
maxdim = parse(Int64, ARGS[2])
beta_max = parse(Float64, ARGS[3])
δt = parse(Float64, ARGS[4])

main(L=L, maxdim=maxdim, beta_max=beta_max, δt=δt)