using MKL
using ITensors
using ITensorTDVP
using CUDA
using Printf
using PyPlot
using HDF5
using LinearAlgebra
using TickTock
include("basis_extend.jl")
include("applyexp.jl")

function solver(H, t, psi0; kwargs...)
    tol_per_unit_time = get(kwargs, :solver_tol, 1E-8)
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

function ITensors.measure!(o::SizeObserver; bond, sweep, half_sweep, psi, kwargs...)
  if bond==1 && half_sweep==2
    # psi_size =  Base.format_bytes(Base.summarysize(psi))
    # println("After sweep $sweep, |psi| = $psi_size")
    GC.gc()
  end
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

function heisenberg(L, J2, Delta, U1, U2, real_evolution)
  os = OpSum()

  # Adding J1 = 1 terms in ladder
  for j in 1:2:(2*L - 3)
    os += Delta, "S1z", j, "S1z", j + 2
    os += 0.5, "S1+", j, "S1-", j + 2
    os += 0.5, "S1-", j, "S1+", j + 2

    os += Delta, "S2z", j, "S2z", j + 2
    os += 0.5, "S2+", j, "S2-", j + 2
    os += 0.5, "S2-", j, "S2+", j + 2

    if (real_evolution)
      # Apply disentangler exp(iHt) on ancilla sites
      os += -1 * Delta, "S1z", j + 1, "S1z", j + 3
      os += -0.5, "S1+", j + 1, "S1-", j + 3
      os += -0.5, "S1-", j + 1, "S1+", j + 3

      os += -1 * Delta, "S2z", j + 1, "S2z", j + 3
      os += -0.5, "S2+", j + 1, "S2-", j + 3
      os += -0.5, "S2-", j + 1, "S2+", j + 3
    end
  end

  # Adding J2 rung terms in ladder
  for j in 1:2:(2*L - 1)
    os += J2, "rung", j

    if (real_evolution)
      # Apply disentangler exp(iHt) on ancilla sites
      os += -1*J2, "rung", j + 1
    end
  end

  # Adding U1 biquadratic terms in ladder
  for j in 1:2:(2*L - 3)
    os += U1, "S1z", j, "S1z", j + 2, "S2z", j, "S2z", j + 2
    os += U1 * 0.5, "S1z", j, "S1z", j + 2, "S2+", j, "S2-", j + 2
    os += U1 * 0.5, "S1z", j, "S1z", j + 2, "S2-", j, "S2+", j + 2

    os += U1 * 0.5, "S1+", j, "S1-", j + 2, "S2z", j, "S2z", j + 2
    os += U1 * 0.25, "S1+", j, "S1-", j + 2, "S2+", j, "S2-", j + 2
    os += U1 * 0.25, "S1+", j, "S1-", j + 2, "S2-", j, "S2+", j + 2

    os += U1 * 0.5, "S1-", j, "S1+", j + 2, "S2z", j, "S2z", j + 2
    os += U1 * 0.25, "S1-", j, "S1+", j + 2, "S2+", j, "S2-", j + 2
    os += U1 * 0.25, "S1-", j, "S1+", j + 2, "S2-", j, "S2+", j + 2

    if (real_evolution)
      # Apply disentangler exp(iHt) on ancilla sites
      os += -U1, "S1z", j + 1, "S1z", j + 3, "S2z", j + 1, "S2z", j + 3
      os += -U1 * 0.5, "S1z", j + 1, "S1z", j + 3, "S2+", j + 1, "S2-", j + 3
      os += -U1 * 0.5, "S1z", j + 1, "S1z", j + 3, "S2-", j + 1, "S2+", j + 3

      os += -U1 * 0.5, "S1+", j + 1, "S1-", j + 3, "S2z", j + 1, "S2z", j + 3
      os += -U1 * 0.25, "S1+", j + 1, "S1-", j + 3, "S2+", j + 1, "S2-", j + 3
      os += -U1 * 0.25, "S1+", j + 1, "S1-", j + 3, "S2-", j + 1, "S2+", j + 3

      os += -U1 * 0.5, "S1-", j + 1, "S1+", j + 3, "S2z", j + 1, "S2z", j + 3
      os += -U1 * 0.25, "S1-", j + 1, "S1+", j + 3, "S2+", j + 1, "S2-", j + 3
      os += -U1 * 0.25, "S1-", j + 1, "S1+", j + 3, "S2-", j + 1, "S2+", j + 3
    end
  end

  # Adding U2 biquadratic terms in ladder
  for j in 1:2:(2*L - 3)
    os += U2, "S1z", j, "S2z", j, "S1z", j + 2, "S2z", j + 2
    os += U2 * 0.5, "S1z", j, "S2z", j, "S1+", j + 2, "S2-", j + 2
    os += U2 * 0.5, "S1z", j, "S2z", j, "S1-", j + 2, "S2+", j + 2

    os += U2 * 0.5, "S1+", j, "S2-", j, "S1z", j + 2, "S2z", j + 2
    os += U2 * 0.25, "S1+", j, "S2-", j, "S1+", j + 2, "S2-", j + 2
    os += U2 * 0.25, "S1+", j, "S2-", j, "S1-", j + 2, "S2+", j + 2

    os += U2 * 0.5, "S1-", j, "S2+", j, "S1z", j + 2, "S2z", j + 2
    os += U2 * 0.25, "S1-", j, "S2+", j, "S1+", j + 2, "S2-", j + 2
    os += U2 * 0.25, "S1-", j, "S2+", j, "S1-", j + 2, "S2+", j + 2

    if (real_evolution)
      # Apply disentangler exp(iHt) on ancilla sites
      os += -U2, "S1z", j + 1, "S2z", j + 1, "S1z", j + 3, "S2z", j + 3
      os += -U2 * 0.5, "S1z", j + 1, "S2z", j + 1, "S1+", j + 3, "S2-", j + 3
      os += -U2 * 0.5, "S1z", j + 1, "S2z", j + 1, "S1-", j + 3, "S2+", j + 3

      os += -U2 * 0.5, "S1+", j + 1, "S2-", j + 1, "S1z", j + 3, "S2z", j + 3
      os += -U2 * 0.25, "S1+", j + 1, "S2-", j + 1, "S1+", j + 3, "S2-", j + 3
      os += -U2 * 0.25, "S1+", j + 1, "S2-", j + 1, "S1-", j + 3, "S2+", j + 3

      os += -U2 * 0.5, "S1-", j + 1, "S2+", j + 1, "S1z", j + 3, "S2z", j + 3
      os += -U2 * 0.25, "S1-", j + 1, "S2+", j + 1, "S1+", j + 3, "S2-", j + 3
      os += -U2 * 0.25, "S1-", j + 1, "S2+", j + 1, "S1-", j + 3, "S2+", j + 3
    end
  end
  
  return os
end

# Measure total magnetization in left half
function magnetization_transfer(L)
  os = OpSum()

  for j in 1:2:(L-1)
    os += -1, "S1z", j
    os += -1, "S2z", j
  end

  return os
end

# Calculating first four moments of full counting statistics of magnetization transfer
function moments(L, ψ, sites, cutoff, maxdim)
  # M_op = cu(MPO(magnetization_transfer(L), sites))
  M_op = cu(MPO(H_dw(L), sites))
  ψ2 = apply(M_op, ψ; cutoff, maxdim)
  M_avg = inner(ψ, ψ2)

  # ψ2 = apply(M_op, ψ2)
  M2 = inner(ψ2, ψ2)

  ψ3 = apply(M_op, ψ2; cutoff, maxdim)
  M3 = inner(ψ2, ψ3)

  # ψ2 = apply(M_op, ψ2)
  M4 = inner(ψ3, ψ3)

  return [M_avg, M2, M3, M4]
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

function main(; L=128, cutoff=1e-16, δτ=0.05, β_max=0.0, δt=0.1, ttotal=100, maxdim=32, J2=0.0, Delta=1.0, U1=0.0, U2=0.0, μ=0.001)
  tick()

  c = div(L,2) + 1 # center site

  filename = "/pscratch/sd/k/kwang98/KPZ/tdvp_coarsegrained_dw_gpu_L$(L)_chi$(maxdim)_beta$(β_max)_dt$(δt)_Jprime$(J2)_U$(U1)_mu$(μ).h5"
  # filename = "/global/scratch/users/kwang98/KPZ/tdvp_coarsegrained_dw_gpu_L$(L)_chi$(maxdim)_beta$(β_max)_dt$(δt)_Jprime$(J2)_U$(U1)_mu$(μ).h5"
  # filename = "tdvp_coarsegrained_dw_gpu_L$(L)_chi$(maxdim)_beta$(β_max)_dt$(δt)_Jprime$(J2)_U$(U1)_mu$(μ).h5"

  if (isfile(filename))
    F = h5open(filename,"r")
    times = read(F, "times")
    Z1s = read(F, "Z1s")
    Z2s = read(F, "Z2s")
    Ss = read(F, "Ss")
    # M_moments = read(F, "M_moments")
    ψ = cu(read(F, "psi", MPS))
    start_time = last(times) + δt
    close(F)

    sites = siteinds(ψ)
    H_real = cu(MPO(heisenberg(L, J2, Delta, U1, U2, true), sites))
  else
    sites = siteinds("S=3/2", 2 * L; conserve_qns=false)
    H_imag = cu(MPO(heisenberg(L, J2, Delta, U1, U2, false), sites))
    H_real = cu(MPO(heisenberg(L, J2, Delta, U1, U2, true), sites))
  
    # Initial state is infinite-temperature mixed state, odd = physical, even = ancilla
    ψ = cu(inf_temp_mps(sites))
    # ψ = basis_extend(ψ, H_real; cutoff, extension_krylovdim=2)
  
    # Create initial domain wall state
    ψ = tdvp(cu(MPO(H_dw(L), sites)), μ, ψ;
        nsweeps=1,
        reverse_step=true,
        normalize=true,
        maxdim=maxdim,
        cutoff=cutoff,
        outputlevel=1,
        nsite=2
      )
    
    # Cool down to inverse temperature 
    for β in δτ:δτ:β_max/2
      @printf("β = %.2f\n", 2*β)
      flush(stdout)
      ψ = tdvp(H_imag, -δτ, ψ;
        nsweeps=1,
        reverse_step=true,
        normalize=true,
        maxdim=maxdim,
        cutoff=cutoff,
        outputlevel=1,
        nsite=2
      )
    end

    times = Float64[]
    Z1s = []
    Z2s = []
    Ss = []
    # M_moments = []
    start_time = δt
  end

  for t in start_time:δt:ttotal
    # Stop simulations before HPC limit to ensure no corruption of data writing
    if peektimer() > (23.5 * 60 * 60)
      break
    end

    # maxlinkdim(ψ) < maxdim ? nsite = 2 : nsite = 1
    # if (maxlinkdim(ψ) < maxdim)
    # @time ψ = basis_extend(ψ, H_real; cutoff, extension_krylovdim=2)
    # end

    ψ = tdvp(solver, H_real, -im * δt, ψ;
      nsweeps=1,
      reverse_step=true,
      normalize=false,
      maxdim=maxdim,
      cutoff=cutoff,
      outputlevel=1,
      nsite=2
    )
    GC.gc()

    Z1 = expect(ψ, "S1z"; sites=1:2:(2*L-1))
    Z2 = expect(ψ, "S2z"; sites=1:2:(2*L-1))
    S = entropy_von_neumann(ITensors.cpu(ψ), L) # Von neumann entropy at half-cut between ancilla and physical (initially unentangled)
    # @time M_moment = moments(L, ψ, sites, cutoff, maxdim)

    println("Time = $t")
    flush(stdout)
    push!(times, t)
    t == δt ? Z1s = Z1 : Z1s = hcat(Z1s, Z1)
    t == δt ? Z2s = Z2 : Z2s = hcat(Z2s, Z2)
    t == δt ? Ss = S : Ss = hcat(Ss, S)
    # t == δt ? M_moments = M_moment : M_moments = hcat(M_moments, M_moment)

    # Writing to data file
    F = h5open(filename,"w")
    F["times"] = times
    F["Z1s"] = Z1s
    F["Z2s"] = Z2s
    F["Ss"] = Ss
    # F["M_moments"] = M_moments
    F["corrs"] = (Z1s[c-1,:] .- Z1s[c,:]) ./ (2 * μ)
    F["psi"] = ITensors.cpu(ψ)
    close(F)

    t≈ttotal && break
  end
end

ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(1)
# ITensors.enable_threaded_blocksparse(true)

L = parse(Int64, ARGS[1])
maxdim = parse(Int64, ARGS[2])
β_max = parse(Float64, ARGS[3])
δt = parse(Float64, ARGS[4])
J2 = parse(Float64, ARGS[5])
U1 = parse(Float64, ARGS[6])
U2 = parse(Float64, ARGS[7])
μ = parse(Float64, ARGS[8])

main(L=L, maxdim=maxdim, β_max=β_max, δt=δt, J2=J2, U1=U1, U2=U2, μ=μ)