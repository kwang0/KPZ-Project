using MKL # Nvidia usage
using CUDA # Nvidia usage
using ITensors
using ITensorMPS
using Printf
using PyPlot
using HDF5
using LinearAlgebra
using TickTock
using Base.Threads

struct SimulationParameters
  L::Int64
  maxdim::Int64
  cutoff::Float64
  δt::Float64
  ttotal::Float64
  J2::Float64
  μ::Float64
end

function solver(H, t, psi0; kwargs...)
    tol_per_unit_time = get(kwargs, :solver_tol, 1f-8)
    solver_kwargs = (;
        maxiter=get(kwargs, :solver_krylovdim, 30),
        outputlevel=get(kwargs, :solver_outputlevel, 0),
    )
    #applyexp tol is absolute, compute from tol_per_unit_time:
    tol = abs(t) * tol_per_unit_time
    psi, info = applyexp(H, t, psi0; tol, solver_kwargs..., kwargs...)
    return psi, info
end

mutable struct SizeObserver <: AbstractObserver
end

function entropy_von_neumann(ψ, b)
  ψ = orthogonalize(ψ, b)
  U,S,V = svd(ψ[b], (linkinds(ψ, b-1)..., siteinds(ψ, b)...))
  SvN = 0.0
  for n=1:dim(S, 1)
    p = S[n,n]^2
    SvN -= p * log(p)
  end
  return SvN
end

function find_lambdas(ψ)
  N = length(ψ)
  Λs = ITensor[]
  sites = siteinds(ψ)

  ϕ = orthogonalize(ψ, 1)
  U,S,V = svd(ϕ[1], (sites[1]))
  # leftind = dag(commonind(U,S))
  # rightind = commonind(V,ϕ[1])
  # linkind = commonind(ψ[1], ψ[2])
  # replaceinds!(V, [rightind], [linkind])
  push!(Λs, S*V)
  for i in 2:(N-1)
    ϕ = orthogonalize(ϕ, i)
    U,S,V = svd(ϕ[i], (commonind(ϕ[i-1], ϕ[i]), sites[i]))
    # leftind = commonind(U,S)
    # rightind = commonind(S,V)
    # linkind = commonind(ψ[i-1], ψ[i])
    # replaceinds!(S, [leftind], [linkind])
    push!(Λs, S*V)
  end

  return Λs
end

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

function ITensors.op(::OpName"U(t)", ::SiteType"S=3/2", s1::Index, s2::Index; t, J2)
  h1 =
    1 / 2 * op("S1+", s1) * op("S1-", s2) +
    1 / 2 * op("S1-", s1) * op("S1+", s2) +
    op("S1z", s1) * op("S1z", s2)
  h2 =
    1 / 2 * op("S2+", s1) * op("S2-", s2) +
    1 / 2 * op("S2-", s1) * op("S2+", s2) +
    op("S2z", s1) * op("S2z", s2)
  rung = op("rung", s1) * op("Id", s2)
  
  return exp(-im * t * ((h1 + h2) + J2 * rung))
end

# Update block only contracting three-site blocks at a time
function updateblock(ψ, sites, i, Λs, W1, W2, cut, m)
  Φ_bar = ψ[i+1] * ψ[i+2] * ψ[i+3]
  W1 = replaceinds(W1, [sites[1],sites[1]',sites[3],sites[3]'], [sites[i],sites[i]',sites[i+2],sites[i+2]'])
  W2 = replaceinds(W2, [sites[2],sites[2]',sites[4],sites[4]'], [sites[i+1],sites[i+1]',sites[i+3],sites[i+3]'])
  Φ_bar = apply(W2, Φ_bar)
  Φ = noprime(Λs[i] * Φ_bar)

  if (i == 1)
    # leftlink = dag(commonind(ψ[i], Φ))
    U,S,V = svd(Φ, (sites[i+1], sites[i+2], commonind(Φ, Λs[i])), cutoff=cut, maxdim=m, righttags="Link,l=$(i+2)")
    ψ[i+3] = V
    Λs[i+2] = S
    Φ_bar = ψ[i] * Φ_bar * dag(V)
    Φ_bar = apply(W1, Φ_bar)
    Φ = Φ_bar

    U,S,V = svd(Φ, (sites[i], sites[i+1]), cutoff=cut, maxdim=m, righttags="Link,l=$(i+1)")
    ψ[i+2] = V
    Λs[i+1] = S
    Φ = U*S
    Φ_bar = Φ_bar * dag(V)

    U,S,V = svd(Φ, (sites[i]), cutoff=cut, maxdim=m, righttags="Link,l=$(i)")
    ψ[i+1] = V
    ψ[i] = Φ_bar * dag(V)
    Λs[i] = S
  else
    # leftlink = dag(commonind(ψ[i], Φ))
    U,S,V = svd(Φ, (sites[i+1], sites[i+2], commonind(Φ, Λs[i])), cutoff=cut, maxdim=m, righttags="Link,l=$(i+2)")
    ψ[i+3] = V
    Λs[i+2] = S
    Φ_bar = ψ[i] * Φ_bar * dag(V)
    Φ_bar = apply(W1, Φ_bar)
    Φ = noprime(Λs[i-1] * Φ_bar)

    # leftlink = dag(commonind(ψ[i-1], Φ))
    U,S,V = svd(Φ, (sites[i], sites[i+1], commonind(Φ, Λs[i-1])), cutoff=cut, maxdim=m, righttags="Link,l=$(i+1)")
    ψ[i+2] = V
    Λs[i+1] = S
    Φ = U*S
    Φ_bar = Φ_bar * dag(V)

    U,S,V = svd(Φ, (sites[i], commonind(Φ, Λs[i-1])), cutoff=cut, maxdim=m, righttags="Link,l=$(i)")
    ψ[i+1] = V
    ψ[i] = Φ_bar * dag(V)
    Λs[i] = S
  end
end

function trotter_sweep(ψ::MPS, sites, Λs::Vector{ITensor}, W1::ITensor, W2::ITensor, cut::Float64, m::Int, even::Bool)
  N = length(ψ)
  start_idx = even ? 1 : 3
  end_idx = even ? N-3 : N-5
  total_blocks = div(end_idx - start_idx, 4) + 1

  # Parallel execution of updateblock!
  @threads for block in 1:total_blocks
    i = start_idx + (block - 1) * 4
    updateblock(ψ, sites, i, Λs, W1, W2, cut, m)
  end
end

function fourth_order_trotter_sweep(ψ, sites, Λs, W1s, W2s, cut, m)
  trotter_sweep(ψ, sites, Λs, W1s[1], W2s[1], cut, m, true)
  trotter_sweep(ψ, sites, Λs, W1s[2], W2s[2], cut, m, false)
  trotter_sweep(ψ, sites, Λs, W1s[3], W2s[3], cut, m, true)
  trotter_sweep(ψ, sites, Λs, W1s[4], W2s[4], cut, m, false)
  trotter_sweep(ψ, sites, Λs, W1s[5], W2s[5], cut, m, true)
  trotter_sweep(ψ, sites, Λs, W1s[6], W2s[6], cut, m, false)
  trotter_sweep(ψ, sites, Λs, W1s[5], W2s[5], cut, m, true)
  trotter_sweep(ψ, sites, Λs, W1s[4], W2s[4], cut, m, false)
  trotter_sweep(ψ, sites, Λs, W1s[3], W2s[3], cut, m, true)
  trotter_sweep(ψ, sites, Λs, W1s[2], W2s[2], cut, m, false)
  trotter_sweep(ψ, sites, Λs, W1s[1], W2s[1], cut, m, true)
end

function create_gate_list(sites, δt, J2)
  a1 = 0.095848502741203681182
  a2 = -0.078111158921637922695
  a3 = 0.5 - (a1 + a2)
  b1 = 0.42652466131587616168
  b2 = -0.12039526945509726545
  b3 = 1 - 2 * (b1 + b2)

  W1s = []
  W2s = []

  push!(W1s, op("U(t)", sites[1], sites[3], t = a1*δt, J2=J2))
  push!(W2s, op("U(t)", sites[2], sites[4], t = -a1*δt, J2=J2))
  push!(W1s, op("U(t)", sites[1], sites[3], t = b1*δt, J2=J2))
  push!(W2s, op("U(t)", sites[2], sites[4], t = -b1*δt, J2=J2))
  push!(W1s, op("U(t)", sites[1], sites[3], t = a2*δt, J2=J2))
  push!(W2s, op("U(t)", sites[2], sites[4], t = -a2*δt, J2=J2))
  push!(W1s, op("U(t)", sites[1], sites[3], t = b2*δt, J2=J2))
  push!(W2s, op("U(t)", sites[2], sites[4], t = -b2*δt, J2=J2))
  push!(W1s, op("U(t)", sites[1], sites[3], t = a3*δt, J2=J2))
  push!(W2s, op("U(t)", sites[2], sites[4], t = -a3*δt, J2=J2))
  push!(W1s, op("U(t)", sites[1], sites[3], t = b3*δt, J2=J2))
  push!(W2s, op("U(t)", sites[2], sites[4], t = -b3*δt, J2=J2))

  return W1s, W2s
end

# Adding "Zeeman terms" to produce domain wall density matrix
function H_dw(L)
  os = OpSum()

  for j in 1:2:(L - 1)
    os += 1, "S1z", j
    os += 1, "S2z", j
  end

  for j in (L+1):2:(2*L - 1)
    os -= 1, "S1z", j
    os -= 1, "S2z", j
  end
  
  return os
end

function main(params::SimulationParameters)
  tick()

  c = div(params.L,2) + 1 # center site

  filename = "/pscratch/sd/k/kwang98/KPZ/production/tebd_coarsegrained_L$(params.L)_chi$(params.maxdim)_Jprime$(params.J2)_mu$(params.μ).h5"

  if (isfile(filename))
    F = h5open(filename,"r")
    times = read(F, "times")
    Z1s = read(F, "Z1s")
    Z2s = read(F, "Z2s")
    Ss = read(F, "Ss")
    ψ = read(F, "psi", MPS)
    ψ_norms = read(F, "psi_norms")
    start_time = last(times) + params.δt
    close(F)

    sites = siteinds(ψ)
    orthogonalize!(ψ, 1)
    Λs = find_lambdas(ψ)
    W1s, W2s = create_gate_list(sites, params.δt, params.J2)
  else
    sites = siteinds("S=3/2", 2 * params.L; conserve_qns=true)
    W1s, W2s = create_gate_list(sites, params.δt, params.J2)
  
    # Initial state is infinite-temperature mixed state, odd = physical, even = ancilla
    ψ = inf_temp_mps(sites)
    # ψ = basis_extend(ψ, H_real; cutoff, extension_krylovdim=2)
  
    # Create initial domain wall state
    ψ = tdvp(MPO(H_dw(params.L), sites), params.μ, ψ;
        nsweeps=1,
        reverse_step=true,
        normalize=true,
        maxdim=params.maxdim,
        cutoff=params.cutoff,
        outputlevel=1,
        nsite=2
      )
    orthogonalize!(ψ, 1)
    Λs = find_lambdas(ψ)

    times = Float64[]
    ψ_norms = Float64[]
    Z1s = []
    Z2s = []
    Ss = []
    start_time = params.δt
  end

  for t in start_time:params.δt:params.ttotal
    # Stop simulations before HPC limit to ensure no corruption of data writing
    if peektimer() > (23.5 * 60 * 60)
      break
    end

    @time fourth_order_trotter_sweep(ψ, sites, Λs, W1s, W2s, params.cutoff, params.maxdim)
    GC.gc()

    Z1 = expect(ψ, "S1z"; sites=1:2:(2*params.L-1))
    Z2 = expect(ψ, "S2z"; sites=1:2:(2*params.L-1))
    S = entropy_von_neumann(ITensors.cpu(ψ), 2*params.L) # Von neumann entropy at half-cut between ancilla and physical (initially unentangled)

    println("Time = $t")
    println("Max bond dimension = $(maxlinkdim(ψ))")
    flush(stdout)
    push!(times, t)
    push!(ψ_norms, norm(ψ))
    t == params.δt ? Z1s = Z1 : Z1s = hcat(Z1s, Z1)
    t == params.δt ? Z2s = Z2 : Z2s = hcat(Z2s, Z2)
    t == params.δt ? Ss = S : Ss = hcat(Ss, S)

    # Writing to data file
    F = h5open(filename,"w")
    F["times"] = times
    F["Z1s"] = Z1s
    F["Z2s"] = Z2s
    F["Ss"] = Ss
    F["corrs"] = (Z1s[c-1,:] .- Z1s[c,:]) ./ (2 * params.μ)
    F["psi"] = ITensors.cpu(ψ)
    F["psi_norms"] = ψ_norms
    close(F)

    t≈params.ttotal && break
  end
end

ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(1)
ITensors.enable_threaded_blocksparse(true)

params = SimulationParameters(
    parse(Int64, ARGS[1]),    # L
    parse(Int64, ARGS[2]),    # maxdim
    1e-12,                     # cutoff
    0.1,  # δt
    100.0,                    # ttotal (or parse from ARGS if it's an input)
    parse(Float64, ARGS[3]),   # J2
    parse(Float64, ARGS[4])   # μ
)

main(params)