using MKL
using ITensors
using ITensorMPS
using CUDA
using Printf
using PyPlot
using HDF5
using LinearAlgebra
using TickTock
# include("basis_extend.jl")
# include("applyexp.jl")

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

function ITensors.op(::OpName"expiSS", ::SiteType"S=1/2", s1::Index, s2::Index; t, x, y, z)
  h =
    0.25 * x * op("X", s1) * op("X", s2) +
    0.25 * y * op("Y", s1) * op("Y", s2) +
    0.25 * z * op("Z", s1) * op("Z", s2)
  
  return cu(exp(-im * t * h))
end

function create_gates(sites, δt, L, offset, rungX, rungY, rungZ)
  op_list = []

  for j in 1:6:(4*L - 3)
    # Only works for offset=5 right now
    if (j == (4*offset - 1))
      push!(op_list, ("expiSS", (j, j + 2), (t=δt, x=rungX, y=rungY, z=rungZ)))
      push!(op_list, ("expiSS", (j + 1, j + 3), (t=-δt, x=rungX, y=rungY, z=rungZ)))
      continue
    end

    push!(op_list, ("expiSS", (j, j + 4), (t=δt, x=1.0, y=1.0, z=1.0)))
    push!(op_list, ("expiSS", (j + 1, j + 5), (t=-δt, x=1.0, y=1.0, z=1.0)))
  end

  for j in 3:6:(4*L - 3)
    if (j == (4*offset - 5))
      push!(op_list, ("expiSS", (j, j + 6), (t=δt, x=1.0, y=1.0, z=1.0)))
      push!(op_list, ("expiSS", (j + 1, j + 7), (t=-δt, x=1.0, y=1.0, z=1.0)))

      push!(op_list, ("expiSS", (j + 2, j + 8), (t=δt, x=1.0, y=1.0, z=1.0)))
      push!(op_list, ("expiSS", (j + 3, j + 9), (t=-δt, x=1.0, y=1.0, z=1.0)))
      continue
    end

    push!(op_list, ("expiSS", (j, j + 4), (t=δt, x=1.0, y=1.0, z=1.0)))
    push!(op_list, ("expiSS", (j + 1, j + 5), (t=-δt, x=1.0, y=1.0, z=1.0)))
  end

  for j in 5:6:(4*L - 3)
    if (j == (4*offset - 3))
      push!(op_list, ("expiSS", (j, j + 2), (t=δt, x=rungX, y=rungY, z=rungZ)))
      push!(op_list, ("expiSS", (j + 1, j + 3), (t=-δt, x=rungX, y=rungY, z=rungZ)))
      continue
    end

    push!(op_list, ("expiSS", (j, j + 4), (t=δt, x=1.0, y=1.0, z=1.0)))
    push!(op_list, ("expiSS", (j + 1, j + 5), (t=-δt, x=1.0, y=1.0, z=1.0)))
  end

  return ops(op_list, sites)
end

function main(; L=128, cutoff=1f-6, δt=0.1, ttotal=20, maxdim=32, offset=5, rungX=0.0, rungY=0.0, rungZ=0.0)
  tick()

  filename = "/pscratch/sd/k/kwang98/KPZ/IBM_sims_L$(L)_chi$(maxdim)_dt$(δt)_offset$(offset)_rungX$(rungX)_rungY$(rungY)_rungZ$(rungZ).h5"
  # filename = "IBM_sims_L$(L)_chi$(maxdim)_dt$(δt)_offset$(offset)_rungX$(rungX)_rungY$(rungY)_rungZ$(rungZ).h5"

  if (isfile(filename))
    F = h5open(filename,"r")
    times = read(F, "times")
    corrs = read(F, "corrs")
    Ss = read(F, "Ss")
    ψ = cu(read(F, "psi", MPS))
    ψ2 = cu(read(F, "psi2", MPS))
    start_time = last(times) + δt
    close(F)

    sites = siteinds(ψ)
    gates = create_gates(sites, δt, L, offset, rungX, rungY, rungZ)
  else
    sites = siteinds("S=1/2", 4 * L + 2; conserve_qns=false)
    gates = create_gates(sites, δt, L, offset, rungX, rungY, rungZ)

    # Initial state is infinite-temperature mixed state, odd = physical, even = ancilla
    ψ = cu(inf_temp_mps(sites))

    orthogonalize!(ψ, 1)
    ψ2 = apply(cu(op("Z",sites[1])), ψ; cutoff, maxdim)

    times = Float64[]
    corrs = []
    Ss = []
    start_time = δt
  end

  for t in start_time:δt:ttotal
    # Stop simulations before HPC limit to ensure no corruption of data writing
    if peektimer() > (23.5 * 60 * 60)
      break
    end

    # if (maxlinkdim(ψ2) < maxdim)
    #   ψ2 = expand(ψ2, H_real; alg="global_krylov", cutoff)
    # end

    ψ2 = apply(gates, ψ2; cutoff, maxdim)
    GC.gc()

    corr = ComplexF64[]
    for i in 1:2:(4*L + 1)
      orthogonalize!(ψ, i)
      orthogonalize!(ψ2, i)
      push!(corr, inner(apply(cu(op("Z",sites[i])), ψ; cutoff, maxdim), ψ2))
    end
    orthogonalize!(ψ2, 1)

    S = entropy_von_neumann(ITensors.cpu(ψ), 2*L) # Von neumann entropy at half-cut between ancilla and physical (initially unentangled)

    println("Time = $t")
    println("Max link dim = $(maxlinkdim(ψ2))")
    flush(stdout)
    push!(times, t)
    t == δt ? corrs = corr : corrs = hcat(corrs, corr)
    t == δt ? Ss = S : Ss = hcat(Ss, S)

    # Writing to data file
    F = h5open(filename,"w")
    F["times"] = times
    F["corrs"] = corrs
    F["psi"] = ITensors.cpu(ψ)
    F["psi2"] = ITensors.cpu(ψ2)
    F["Ss"] = Ss
    close(F)

    t≈ttotal && break
  end
end

ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(1)
# ITensors.enable_threaded_blocksparse(true)

L = parse(Int64, ARGS[1])
maxdim = parse(Int64, ARGS[2])
δt = parse(Float64, ARGS[3])
offset = parse(Int64, ARGS[4])
rungX = parse(Float64, ARGS[5])
rungY = parse(Float64, ARGS[6])
rungZ = parse(Float64, ARGS[7])

main(L=L, maxdim=maxdim, δt=δt, offset=offset, rungX=rungX, rungY=rungY, rungZ=rungZ)