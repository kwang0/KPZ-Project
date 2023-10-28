using MKL
using LinearAlgebra
using ITensors
# using ITensorGPU
using Printf
using PyPlot
using HDF5
using TickTock

function inf_temp_mps(sites)
  num_sites = length(sites)
  if (num_sites % 2 != 0)
    throw(DomainError(num_sites,"Expects even number of sites for ancilla-physical singlets."))
  else
    state = ["UpDn" for n=1:num_sites]
    ψ = MPS(sites, state)
    for j = 1:2:num_sites-1
      s1 = sites[j]
      s2 = sites[j+1]
          
      if(j == 1)
        rightlink = commonind(ψ[j+1],ψ[j+2])
        A = ITensor(ComplexF64, s1, s2, rightlink)

        A[s1=>1, s2=>4, rightlink => 1] = 1/2
        A[s1=>4, s2=>1, rightlink => 1] = 1/2
        A[s1=>2, s2=>3, rightlink => 1] = 1/2
        A[s1=>3, s2=>2, rightlink => 1] = 1/2

        U,S,V = svd(A, (s1), cutoff=1e-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V

      elseif (j == num_sites-1)
        leftlink = dag(commonind(ψ[j-1], ψ[j]))
        A = ITensor(ComplexF64, s1, s2, leftlink)

        A[s1=>1, s2=>4, leftlink => 1] = 1/2
        A[s1=>4, s2=>1, leftlink => 1] = 1/2
        A[s1=>2, s2=>3, leftlink => 1] = 1/2
        A[s1=>3, s2=>2, leftlink => 1] = 1/2

        U,S,V = svd(A, (s1, leftlink), cutoff=1e-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V
        
      else
        rightlink = commonind(ψ[j+1], ψ[j+2])
        leftlink = dag(commonind(ψ[j-1], ψ[j]))
    
        A = ITensor(ComplexF64, s1, s2, rightlink, leftlink)

        A[s1=>1, s2=>4, rightlink=>1, leftlink => 1] = 1/2
        A[s1=>4, s2=>1, rightlink=>1, leftlink => 1] = 1/2
        A[s1=>2, s2=>3, rightlink=>1, leftlink => 1] = 1/2
        A[s1=>3, s2=>2, rightlink=>1, leftlink => 1] = 1/2

        U,S,V = svd(A, (s1, leftlink), cutoff=1e-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V
      end
    end

    return ψ
  end
end

# Representation of two spin-1/2's coarse-grained onto one spin-3/2 Hilbert space
# Convention is (|up,up>, |up,down>, |down,up>, |down,down>)
function ITensors.space(::SiteType"S=3/2";
  conserve_qns=false)
  if conserve_qns
    return [QN("Sz",1)=>1,QN("Sz",0)=>2,QN("Sz",-1)=>1]
  end
  return 4
end

ITensors.state(::StateName"UpUp", ::SiteType"S=3/2") = [1.0, 0, 0, 0]
ITensors.state(::StateName"UpDn", ::SiteType"S=3/2") = [0, 1.0, 0, 0]
ITensors.state(::StateName"DnUp", ::SiteType"S=3/2") = [0, 0, 1.0, 0]
ITensors.state(::StateName"DnDn", ::SiteType"S=3/2") = [0, 0, 0, 1.0]

ITensors.op(::OpName"S1z",::SiteType"S=3/2") =
  [+1/2   0    0    0
     0  +1/2   0    0 
     0    0  -1/2   0
     0    0    0  -1/2]
     
ITensors.op(::OpName"S2z",::SiteType"S=3/2") =
  [+1/2   0    0    0
   0  -1/2   0    0 
   0    0  +1/2   0
   0    0    0  -1/2]

ITensors.op(::OpName"S1+",::SiteType"S=3/2") =
  [0   0  1  0
   0   0  0  1
   0   0  0  0
   0   0  0  0] 

ITensors.op(::OpName"S2+",::SiteType"S=3/2") =
  [0   1  0  0
   0   0  0  0
   0   0  0  1
   0   0  0  0] 

ITensors.op(::OpName"S1-",::SiteType"S=3/2") =
  [0   0  0   0
   0   0  0   0
   1   0  0   0
   0   1  0  0]

ITensors.op(::OpName"S2-",::SiteType"S=3/2") =
  [0   0  0   0
   1   0  0   0
   0   0  0   0
   0   0  1  0]
ITensors.op(::OpName"rung",::SiteType"S=3/2") =
   [1/4   0     0     0
    0    -1/4   1/2   0
    0     1/2   -1/4   0
    0     0     0    1/4]
ITensors.op(::OpName"Id",::SiteType"S=3/2") =
  [1   0  0   0
   0   1  0   0
   0   0  1   0
   0   0  0  1]

function ITensors.op(::OpName"expiSS", ::SiteType"S=3/2", s1::Index, s2::Index; t, J1, J2)
  h1 =
    1 / 2 * op("S1+", s1) * op("S1-", s2) +
    1 / 2 * op("S1-", s1) * op("S1+", s2) +
    op("S1z", s1) * op("S1z", s2)
  h2 =
    1 / 2 * op("S2+", s1) * op("S2-", s2) +
    1 / 2 * op("S2-", s1) * op("S2+", s2) +
    op("S2z", s1) * op("S2z", s2)
  rung = op("rung", s1) * op("Id", s2)
  
  return exp(-im * t * (J1 * (h1 + h2) + J2 * rung))
end

function create_gates(L, sites, t, J1, J2)
  gates = ITensor[]
  for i in 1:2:(2*L-3)
    push!(gates, op("expiSS", sites[i], sites[i+2], t=t, J1=J1, J2=J2))
    push!(gates, op("expiSS", sites[i+1], sites[i+3], t=-t, J1=J1, J2=J2))
    # push!(gates, cuITensor(op("expiSS", sites[i], sites[i+2], t=t)))
    # push!(gates, cuITensor(op("expiSS", sites[i+1], sites[i+3], t=-t)))
  end

  return gates
end

function create_gate_list(L, sites, t, J1, J2)
  a1 = 0.095848502741203681182
  a2 = -0.078111158921637922695
  a3 = 0.5 - (a1 + a2)
  b1 = 0.42652466131587616168
  b2 = -0.12039526945509726545
  b3 = 1 - 2 * (b1 + b2)

  gate_list = Vector{Vector{ITensor}}(undef,6)

  gate_list[1] = create_gates(L, sites, a1 * t, J1, J2)
  gate_list[2] = create_gates(L, sites, b1 * t, J1, J2)
  gate_list[3] = create_gates(L, sites, a2 * t, J1, J2)
  gate_list[4] = create_gates(L, sites, b2 * t, J1, J2)
  gate_list[5] = create_gates(L, sites, a3 * t, J1, J2)
  gate_list[6] = create_gates(L, sites, b3 * t, J1, J2)

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
  alg = "divide_and_conquer"

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

function main(; L=128, cutoff=1E-12, δτ=0.05, β_max=3.0, δt=0.1, ttotal=100, maxdim=32, J1=1.0, J2=0.0)
  tick()

  # Cool down to inverse temperature 
  # for β in δτ:δτ:β_max/2
  #   @printf("β = %.2f\n", 2*β)
  #   flush(stdout)
  #   ψ = trotter_sweep(ψ, L, sites, (1/2) * δτ, false, cutoff, maxdim)
  #   # ψ = trotter_sweep(ψ, L, s, ((1-2*α)/2) * δτ, false, cutoff, maxdim)
  #   # ψ = trotter_sweep(ψ, L, s, (α/2) * δτ, false, cutoff, maxdim)
  #   normalize!(ψ)
  # end

  c = L - 1 # center site

  filename = "/pscratch/sd/k/kwang98/KPZ/tebd_coarsegrained_contract_L$(L)_chi$(maxdim)_beta$(β_max)_dt$(δt)_Jprime$(J2).h5"
  # filename = "tebd_coarsegrained_contract_L$(L)_chi$(maxdim)_beta$(β_max)_dt$(δt)_Jprime$(J2).h5"
  if (isfile(filename))
    F = h5open(filename,"r")
    times = read(F, "times")
    corrs = read(F, "corrs")
    ψ = read(F, "psi", MPS)
    ψ2 = read(F, "psi2", MPS)
    ψ_norms = read(F, "psi_norms")
    ψ2_norms = read(F, "psi2_norms")
    start_time = last(times)
    close(F)

    sites = siteinds(ψ)
    Sz_center = op("S1z",sites[c])
    # Make real-time evolution gates
    real_gate_list = create_gate_list(L, sites, δt, J1, J2)
    GC.gc()
  else
    sites = siteinds("S=3/2", 2 * L; conserve_qns=true)

    # Initial state is infinite-temperature mixed state (purification)
    ψ = inf_temp_mps(sites)
    real_gate_list = create_gate_list(L, sites, δt, J1, J2)
    GC.gc()
    
    Sz_center = op("S1z",sites[c])
    orthogonalize!(ψ, c)
    ψ2 = apply(2 * Sz_center, ψ; cutoff, maxdim)
    # normalize!(ψ2)
  
    times = Float64[]
    corrs = []
    ψ_norms = Float64[]
    ψ2_norms = Float64[]
    start_time = 0.0
  end

  for t in start_time:δt:ttotal
    println(maxlinkdim(ψ))
    println(maxlinkdim(ψ2))

    corr = ComplexF64[]
    for i in 1:2:(2*L - 1)
      orthogonalize!(ψ, i)
      orthogonalize!(ψ2, i)
      S1z = 2 * op("S1z",sites[i])
      S2z = 2 * op("S2z",sites[i])
      push!(corr, inner(apply(S1z, ψ; cutoff, maxdim), ψ2))
      push!(corr, inner(apply(S2z, ψ; cutoff, maxdim), ψ2))
    end
    orthogonalize!(ψ2, c)
    println("Time = $t")
    println(corr[c])
    flush(stdout)
    push!(times, t)
    t == 0.0 ? corrs = corr : corrs = hcat(corrs, corr)
    push!(ψ_norms, norm(ψ))
    push!(ψ2_norms, norm(ψ2))
    
    # Writing to data file
    F = h5open(filename,"w")
    F["times"] = times
    F["corrs"] = corrs
    F["psi"] = ψ
    F["psi2"] = ψ2
    F["psi_norms"] = ψ_norms
    F["psi2_norms"] = ψ2_norms
    close(F)

    t≈ttotal && break
        
    # Stop simulations before HPC limit to ensure no corruption of data writing
    if peektimer() > (23.5 * 60 * 60)
      break
    end

    @time ψ = fourth_order_contract(ψ, L, sites, real_gate_list, cutoff, maxdim)
    GC.gc()
    # normalize!(ψ)
    @time ψ2 = fourth_order_contract(ψ2, L, sites, real_gate_list, cutoff, maxdim)
    GC.gc()
    # normalize!(ψ2)

    println("Max bond dimension is $(maxlinkdim(ψ2))")
    println("Norm of psi2 is $(norm(ψ2))")
  end

  return times, corrs
end

# Set to identity to run on CPU
# gpu = cu

ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(256)
# ITensors.enable_threaded_blocksparse(true)

L = parse(Int64, ARGS[1])
maxdim = parse(Int64, ARGS[2])
β_max = parse(Float64, ARGS[3])
δt = parse(Float64, ARGS[4])
J2 = parse(Float64, ARGS[5])

main(L=L, maxdim=maxdim, β_max=β_max, δt=δt, J2=J2)