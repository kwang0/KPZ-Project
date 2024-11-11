using MKL
using ITensors
using ITensorTDVP
# using CUDA
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
  cutoff::Float64
  β_max::Float64
  δt::Float64
  ttotal::Float64
  δτ::Float64
  U::Float64
  μ::Float64
end

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
    state = [isodd(n) ? "1" : "4" for n=1:num_sites]
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

# Representation of two spin-1/2's coarse-grained onto four-dimensional Hilbert space
# Convention is (|up,up>, |up,down>, |down,up>, |down,down>)
function ITensors.space(::SiteType"S=3/2";
  conserve_qns=false)
  if conserve_qns
    return [
      QN(("Q1", -1), ("Q2", 0), ("Q3", 0)) => 1
      QN(("Q1", 0), ("Q2", 1), ("Q3", -1)) => 1
      QN(("Q1", 0), ("Q2", 0), ("Q3", 1)) => 1
      QN(("Q1", 1), ("Q2", 1), ("Q3", 0)) => 1
    ]
    # return [QN("Sz",1)=>1,QN("Sz",0)=>2,QN("Sz",-1)=>1]
  end
  return 4
end

ITensors.state(::StateName"1", ::SiteType"S=3/2") = [1.0, 0, 0, 0]
ITensors.state(::StateName"2", ::SiteType"S=3/2") = [0, 1.0, 0, 0]
ITensors.state(::StateName"3", ::SiteType"S=3/2") = [0, 0, 1.0, 0]
ITensors.state(::StateName"4", ::SiteType"S=3/2") = [0, 0, 0, 1.0]

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

# Define ITensors operators by |i><j|
N_hilbert = 4
for i in 1:N_hilbert
  for j in 1:N_hilbert
    op_name = Symbol("S_$(i)_$(j)")

    @eval begin
      ITensors.op(::OpName{$(QuoteNode(op_name))}, ::SiteType"S=3/2") =
      let matrix = zeros($(N_hilbert), $(N_hilbert))
        matrix[$i, $j] = 1.0
        matrix
      end
    end
  end
end

function hamiltonian(L, U, real_evolution)
  os = OpSum()

  for n in 1:2:(4*L - 5)
    for i in 1:4
      for j in 1:4
        # Adding P terms
        os += 1, "S_$(i)_$(j)", n, "S_$(j)_$(i)", n + 4
        # if j != i
        #   os += 1, "S_$(j)_$(i)", n, "S_$(i)_$(j)", n + 4
        # end

        # Apply disentangler exp(iHt) on ancilla sites
        if (real_evolution)
          os += -1, "S_$(i)_$(j)", n + 1, "S_$(j)_$(i)", n + 5
          # if j != i
          #   os += -1, "S_$(j)_$(i)", n + 1, "S_$(i)_$(j)", n + 5
          # end
        end

        # Adding U terms
        if ((n - 1) % 4) == 0
          for k in 1:4
            for l in 1:4
              os += U, "S_$(i)_$(j)", n, "S_$(j)_$(i)", n + 4, "S_$(k)_$(l)", n + 2, "S_$(l)_$(k)", n + 6

              # Apply disentangler exp(iHt) on ancilla sites
              if (real_evolution)
                os += -U, "S_$(i)_$(j)", n + 1, "S_$(j)_$(i)", n + 5, "S_$(k)_$(l)", n + 3, "S_$(l)_$(k)", n + 7
              end

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

  for j in 1:2:(2*L-1)
    os += -1, "S1z", j
    os += -1, "S2z", j
  end

  return os
end

# Adding "Zeeman terms" to produce domain wall density matrix
function H_dw(L)
  os = OpSum()

  for j in 1:2:(2*L - 1)
    os += 1, "S1z", j
    os += 1, "S2z", j
  end

  for j in (2*L+1):2:(4*L - 1)
    os -= 1, "S1z", j
    os -= 1, "S2z", j
  end
  
  return os
end

function main(params::SimulationParameters)
  tick()

  c = params.L + 1 # center site

  filename = "/pscratch/sd/k/kwang98/KPZ/tdvp_su(4)_dw_L$(params.L)_chi$(params.maxdim)_beta$(params.β_max)_dt$(params.δt)_U$(params.U)_mu$(params.μ)_conserve.h5"
  # filename = "/global/scratch/users/kwang98/KPZ/tdvp_su(4)_dw_L$(params.L)_chi$(params.maxdim)_beta$(params.β_max)_dt$(params.δt)_U$(params.U)_mu$(params.μ).h5"
  # filename = "tdvp_su(4)_dw_L$(params.L)_chi$(params.maxdim)_beta$(params.β_max)_dt$(params.δt)_U$(params.U)_mu$(params.μ).h5"

  if (isfile(filename))
    F = h5open(filename,"r")
    times = read(F, "times")
    Z1s = read(F, "Z1s")
    Z2s = read(F, "Z2s")
    Ss = read(F, "Ss")
    ψ = read(F, "psi", MPS)
    start_time = last(times) + params.δt
    close(F)

    sites = siteinds(ψ)
    H_real_cpu = MPO(hamiltonian(params.L, params.U, true), sites)
    H_real = H_real_cpu
  else
    sites = siteinds("S=3/2", 4 * params.L; conserve_qns=true)
    H_imag = MPO(hamiltonian(params.L, params.U, false), sites)
    H_real = MPO(hamiltonian(params.L, params.U, true), sites)
  
    # Initial state is infinite-temperature mixed state, odd = physical, even = ancilla
    ψ = inf_temp_mps(sites)
  
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
    
    # println(params.ttotal)
    # println(params.δt)
    # println(params.δτ)
    # Cool down to inverse temperature 
    # for β in δτ:δτ:β_max/2
    #   @printf("β = %.2f\n", 2*β)
    #   flush(stdout)
    #   ψ = tdvp(H_imag, -δτ, ψ;
    #     nsweeps=1,
    #     reverse_step=true,
    #     normalize=true,
    #     maxdim=params.maxdim,
    #     cutoff=params.cutoff,
    #     outputlevel=1,
    #     nsite=2
    #   )
    # end

    times = Float64[]
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

    # maxlinkdim(ψ) < maxdim ? nsite = 2 : nsite = 1
    # if (maxlinkdim(ψ) < params.maxdim)
    if (maxlinkdim(ψ) == 4)
      # @time ψ = basis_extend(ψ, H_real_cpu; cutoff, extension_krylovdim=2)
      @time ψ = expand(ψ, H_real; alg="global_krylov", params.cutoff, krylovdim=10)
    end

    ψ = tdvp(H_real, -im * params.δt, ψ;
      nsweeps=1,
      reverse_step=true,
      normalize=false,
      maxdim=params.maxdim,
      cutoff=params.cutoff,
      outputlevel=1,
      nsite=2
    )
    GC.gc()

    Z1 = expect(ψ, "S1z"; sites=1:2:(4*params.L-1))
    Z2 = expect(ψ, "S2z"; sites=1:2:(4*params.L-1))
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
ITensors.enable_threaded_blocksparse(true)

params = SimulationParameters(
    parse(Int64, ARGS[1]),    # L
    parse(Int64, ARGS[2]),    # maxdim
    1e-16,                    # cutoff
    parse(Float64, ARGS[3]),  # β_max
    parse(Float64, ARGS[4]),  # δt
    100.0,                    # ttotal (or parse from ARGS if it's an input)
    0.05,                     # δτ (or parse from ARGS if it's an input)
    parse(Float64, ARGS[5]),  # U
    parse(Float64, ARGS[6])   # μ
)

# L = parse(Int64, ARGS[1])
# maxdim = parse(Int64, ARGS[2])
# β_max = parse(Float64, ARGS[3])
# δt = parse(Float64, ARGS[4])
# U = parse(Float64, ARGS[5])
# μ = parse(Float64, ARGS[6])

# main(L=L, maxdim=maxdim, β_max=β_max, δt=δt, U=U, μ=μ)
main(params)