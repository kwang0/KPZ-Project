using MKL # Nvidia usage
using CUDA # Nvidia usage
# using Mtl # Mac usage
# cu = mtl # Mac usage
using ITensors
using ITensorTDVP
using Printf
using PyPlot
using HDF5
using LinearAlgebra
using TickTock
include("basis_extend.jl")
include("applyexp.jl")
# include("expand.jl")

struct SimulationParameters
  L::Int64
  maxdim::Int64
  cutoff::Float32
  β_max::Float32
  δt::Float32
  ttotal::Float32
  δτ::Float32
  U::Float32
  μ::Float32
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
    state = ["22" for n=1:num_sites]
    ψ = MPS(sites, state)
    for j = 1:2:num_sites-1
      s1 = sites[j]
      s2 = sites[j+1]
          
      if(j == 1)
        rightlink = commonind(ψ[j+1],ψ[j+2])
        A = ITensor(ComplexF32, s1, s2, rightlink)

        for i in 1:9
          A[s1=>i, s2=>(10-i), rightlink => 1] = 1/3
        end

        U,S,V = svd(A, (s1), cutoff=1f-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V

      elseif (j == num_sites-1)
        leftlink = dag(commonind(ψ[j-1], ψ[j]))
        A = ITensor(ComplexF32, s1, s2, leftlink)

        for i in 1:9
          A[s1=>i, s2=>(10-i), leftlink => 1] = 1/3
        end

        U,S,V = svd(A, (s1, leftlink), cutoff=1f-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V
        
      else
        rightlink = commonind(ψ[j+1], ψ[j+2])
        leftlink = dag(commonind(ψ[j-1], ψ[j]))
    
        A = ITensor(ComplexF32, s1, s2, rightlink, leftlink)

        for i in 1:9
          A[s1=>i, s2=>(10-i), rightlink=>1, leftlink => 1] = 1/3
        end

        U,S,V = svd(A, (s1, leftlink), cutoff=1f-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V
      end
    end

    return ψ
  end
end

# 9-dimensional coarsegrained Hilbert space
function ITensors.space(::SiteType"SU(3)";
  conserve_qns=false)
  # if conserve_qns
  #   return [
  #     QN(("Q1", 1), ("Q2", 0)) => 1
  #     QN(("Q1", 0), ("Q2", 1)) => 1
  #     QN(("Q1", -1), ("Q2", -1)) => 1
  #   ]
  # end
  return 9
end

ITensors.state(::StateName"11", ::SiteType"SU(3)") = [1.0, 0, 0, 0, 0, 0, 0, 0, 0]
ITensors.state(::StateName"12", ::SiteType"SU(3)") = [0, 1.0, 0, 0, 0, 0, 0, 0, 0]
ITensors.state(::StateName"13", ::SiteType"SU(3)") = [0, 0, 1.0, 0, 0, 0, 0, 0, 0]
ITensors.state(::StateName"21", ::SiteType"SU(3)") = [0, 0, 0, 1.0, 0, 0, 0, 0, 0]
ITensors.state(::StateName"22", ::SiteType"SU(3)") = [0, 0, 0, 0, 1.0, 0, 0, 0, 0]
ITensors.state(::StateName"23", ::SiteType"SU(3)") = [0, 0, 0, 0, 0, 1.0, 0, 0, 0]
ITensors.state(::StateName"31", ::SiteType"SU(3)") = [0, 0, 0, 0, 0, 0, 1.0, 0, 0]
ITensors.state(::StateName"32", ::SiteType"SU(3)") = [0, 0, 0, 0, 0, 0, 0, 1.0, 0]
ITensors.state(::StateName"33", ::SiteType"SU(3)") = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]

Sz = 
[1   0   0
0   0   0
0   0  -1]
Id = 
[1   0   0
0   1   0
0   0  1]

ITensors.op(::OpName"S1z",::SiteType"SU(3)") =
kron(Sz, Id)

ITensors.op(::OpName"S2z",::SiteType"SU(3)") =
kron(Id, Sz)

# Define ITensors operators by |i><j|
N_hilbert = 3
for i in 1:N_hilbert
  for j in 1:N_hilbert
    op_name = Symbol("S1_$(i)_$(j)")

    @eval begin
      ITensors.op(::OpName{$(QuoteNode(op_name))}, ::SiteType"SU(3)") =
      let matrix = zeros($(N_hilbert^2), $(N_hilbert^2))
        matrix1 = zeros($(N_hilbert), $(N_hilbert))
        matrix1[$i, $j] = 1.0
        matrix = kron(matrix1, Id)
        matrix
      end
    end

    op_name = Symbol("S2_$(i)_$(j)")

    @eval begin
      ITensors.op(::OpName{$(QuoteNode(op_name))}, ::SiteType"SU(3)") =
      let matrix = zeros($(N_hilbert^2), $(N_hilbert^2))
        matrix1 = zeros($(N_hilbert), $(N_hilbert))
        matrix1[$i, $j] = 1.0
        matrix = kron(Id, matrix1)
        matrix
      end
    end
  end
end

function hamiltonian(L, U, real_evolution)
  os = OpSum()

  for n in 1:2:(2*L - 3)
    
    # Adding P terms
    for i in 1:3
      for j in 1:3
        os += 1.0, "S1_$(i)_$(j)", n, "S1_$(j)_$(i)", n + 2
        os += 1.0, "S2_$(i)_$(j)", n, "S2_$(j)_$(i)", n + 2

        # Apply disentangler exp(iHt) on ancilla sites
        if (real_evolution)
          os += -1.0, "S1_$(i)_$(j)", n + 1, "S1_$(j)_$(i)", n + 3
          os += -1.0, "S2_$(i)_$(j)", n + 1, "S2_$(j)_$(i)", n + 3
        end

        for k in 1:3
          for l in 1:3
            os += U, "S1_$(i)_$(j)", n, "S1_$(j)_$(i)", n + 2, "S2_$(k)_$(l)", n, "S2_$(l)_$(k)", n + 2

            # Apply disentangler exp(iHt) on ancilla sites
            if (real_evolution)
              os += -U, "S1_$(i)_$(j)", n + 1, "S1_$(j)_$(i)", n + 3, "S2_$(k)_$(l)", n + 1, "S2_$(l)_$(k)", n + 3
            end

          end
        end
      end
    end
  end

  return os
end

# U perturbations
function hamiltonian_U(L, U, real_evolution)
  os = OpSum()

  for n in 1:2:(2*L - 3)
    for i in 1:3
      for j in 1:3
        for k in 1:3
          for l in 1:3
            os += U, "S1_$(i)_$(j)", n, "S1_$(j)_$(i)", n + 2, "S2_$(k)_$(l)", n, "S2_$(l)_$(k)", n + 2

            # Apply disentangler exp(iHt) on ancilla sites
            if (real_evolution)
              os += -U, "S1_$(i)_$(j)", n + 1, "S1_$(j)_$(i)", n + 3, "S2_$(k)_$(l)", n + 1, "S2_$(l)_$(k)", n + 3
            end

          end
        end
      end
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

  filename = "/pscratch/sd/k/kwang98/KPZ/tdvp_su(3)_dw_gpu_L$(params.L)_chi$(params.maxdim)_beta$(params.β_max)_dt$(params.δt)_U$(params.U)_mu$(params.μ).h5"
  # filename = "/global/scratch/users/kwang98/KPZ/tdvp_su(3)_dw_gpu_L$(params.L)_chi$(params.maxdim)_beta$(params.β_max)_dt$(params.δt)_U$(params.U)_mu$(params.μ).h5"
  # filename = "tdvp_su(3)_dw_gpu_L$(params.L)_chi$(params.maxdim)_beta$(params.β_max)_dt$(params.δt)_U$(params.U)_mu$(params.μ).h5"

  if (isfile(filename))
    F = h5open(filename,"r")
    times = read(F, "times")
    Z1s = read(F, "Z1s")
    Z2s = read(F, "Z2s")
    Ss = read(F, "Ss")
    ψ = cu(read(F, "psi", MPS))
    start_time = last(times) + params.δt
    close(F)

    sites = siteinds(ψ)
    H_real = cu(MPO(hamiltonian(params.L, params.U, true), sites))
    # H_real_U = cu(MPO(hamiltonian_U(params.L, params.U, true), sites))
  else
    sites = siteinds("SU(3)", 2 * params.L; conserve_qns=false)
    H_real = cu(MPO(hamiltonian(params.L, params.U, true), sites))
    # H_real_U = cu(MPO(hamiltonian_U(params.L, params.U, false), sites))
  
    # Initial state is infinite-temperature mixed state, odd = physical, even = ancilla
    ψ = cu(inf_temp_mps(sites))
    # ψ = basis_extend(ψ, H_real; cutoff, extension_krylovdim=2)
  
    # Create initial domain wall state
    ψ = tdvp(cu(MPO(H_dw(params.L), sites)), params.μ, ψ;
        nsweeps=1,
        reverse_step=true,
        normalize=true,
        maxdim=params.maxdim,
        cutoff=params.cutoff,
        outputlevel=1,
        nsite=1
      )

    times = Float32[]
    Z1s = []
    Z2s = []
    Ss = []
    # M_moments = []
    start_time = params.δt
  end

  for t in start_time:params.δt:params.ttotal
    # Stop simulations before HPC limit to ensure no corruption of data writing
    if peektimer() > (23.5 * 60 * 60)
      break
    end

    # ψ = ITensors.cpu(ψ)
    # H_cpu = ITensors.cpu(H_real)
    # maxlinkdim(ψ) < maxdim ? nsite = 2 : nsite = 1
    # if (maxlinkdim(ψ) < params.maxdim)
    # if (maxlinkdim(ψ) == 9)
      # @time ψ = basis_extend(ψ, H_real_cpu; cutoff, extension_krylovdim=2)
    #   @time ψ = expand(ψ, H_real_cpu; alg="global_krylov", params.cutoff, krylovdim=2)
    # end
    # ψ = cu(ψ)

    numsite = (linkdims(ψ)[3] == params.maxdim) ? 1 : 2

    ψ = tdvp(H_real, -im * params.δt, ψ;
      nsweeps=1,
      reverse_step=true,
      normalize=false,
      maxdim=params.maxdim,
      cutoff=params.cutoff,
      outputlevel=1,
      nsite=numsite
    )
    GC.gc()

    Z1 = expect(ψ, "S1z"; sites=1:2:(2*params.L-1))
    Z2 = expect(ψ, "S2z"; sites=1:2:(2*params.L-1))
    S = entropy_von_neumann(ITensors.cpu(ψ), 2*params.L) # Von neumann entropy at half-cut between ancilla and physical (initially unentangled)
    # @time M_moment = moments(L, ψ, sites, cutoff, maxdim)

    println("Time = $t")
    flush(stdout)
    push!(times, t)
    t == params.δt ? Z1s = Z1 : Z1s = hcat(Z1s, Z1)
    t == params.δt ? Z2s = Z2 : Z2s = hcat(Z2s, Z2)
    t == params.δt ? Ss = S : Ss = hcat(Ss, S)
    # t == δt ? M_moments = M_moment : M_moments = hcat(M_moments, M_moment)

    # Writing to data file
    F = h5open(filename,"w")
    F["times"] = times
    F["Z1s"] = Z1s
    F["Z2s"] = Z2s
    F["Ss"] = Ss
    # F["M_moments"] = M_moments
    F["corrs"] = (Z1s[c-1,:] .- Z1s[c,:]) ./ (2 * params.μ)
    F["psi"] = ITensors.cpu(ψ)
    close(F)

    t≈params.ttotal && break
  end
end

ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(1)
# ITensors.enable_threaded_blocksparse(true)

params = SimulationParameters(
    parse(Int64, ARGS[1]),    # L
    parse(Int64, ARGS[2]),    # maxdim
    1f-16,                     # cutoff
    parse(Float32, ARGS[3]),  # β_max
    parse(Float32, ARGS[4]),  # δt
    100.0,                    # ttotal (or parse from ARGS if it's an input)
    0.05,                     # δτ (or parse from ARGS if it's an input)
    parse(Float32, ARGS[5]),  # U
    parse(Float32, ARGS[6])   # μ
)

# L = parse(Int64, ARGS[1])
# maxdim = parse(Int64, ARGS[2])
# β_max = parse(Float64, ARGS[3])
# δt = parse(Float64, ARGS[4])
# U = parse(Float64, ARGS[5])
# μ = parse(Float64, ARGS[6])

# main(L=L, maxdim=maxdim, β_max=β_max, δt=δt, U=U, μ=μ)
main(params)