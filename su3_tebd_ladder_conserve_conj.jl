using ITensors
using ITensorTDVP
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
  β_max::Float64
  δt::Float64
  ttotal::Float64
  δτ::Float64
  U::Float64
  μ::Float64
end

function solver(H, t, psi0; kwargs...)
    tol_per_unit_time = get(kwargs, :solver_tol, 1e-8)
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
        A = ITensor(ComplexF64, s1, s2, rightlink)

        for i in 1:9
          A[s1=>i, s2=>i, rightlink => 1] = 1/3
        end

        U,S,V = svd(A, (s1), cutoff=1e-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V

      elseif (j == num_sites-1)
        leftlink = dag(commonind(ψ[j-1], ψ[j]))
        A = ITensor(ComplexF64, s1, s2, leftlink)

        for i in 1:9
          A[s1=>i, s2=>i, leftlink => 1] = 1/3
        end

        U,S,V = svd(A, (s1, leftlink), cutoff=1e-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V
        
      else
        rightlink = commonind(ψ[j+1], ψ[j+2])
        leftlink = dag(commonind(ψ[j-1], ψ[j]))
    
        A = ITensor(ComplexF64, s1, s2, rightlink, leftlink)

        for i in 1:9
          A[s1=>i, s2=>i, rightlink=>1, leftlink => 1] = 1/3
        end

        U,S,V = svd(A, (s1, leftlink), cutoff=1e-16, lefttags="Link,l=$(j)")
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
  if conserve_qns
    return [
      QN(("Q1", 2), ("Q2", 2)) => 1
      QN(("Q1", 1), ("Q2", 0)) => 1
      QN(("Q1", 0), ("Q2", 1)) => 1
      QN(("Q1", 1), ("Q2", 0)) => 1
      QN(("Q1", 0), ("Q2", -2)) => 1
      QN(("Q1", -1), ("Q2", -1)) => 1
      QN(("Q1", 0), ("Q2", 1)) => 1
      QN(("Q1", -1), ("Q2", -1)) => 1
      QN(("Q1", -2), ("Q2", 0)) => 1
    ]
  end
  return 9
end

# 9-dimensional coarsegrained Hilbert space
function ITensors.space(::SiteType"SU(3)_conj";
  conserve_qns=false)
  if conserve_qns
    return [
      QN(("Q1", -2), ("Q2", -2)) => 1
      QN(("Q1", -1), ("Q2", 0)) => 1
      QN(("Q1", 0), ("Q2", -1)) => 1
      QN(("Q1", -1), ("Q2", 0)) => 1
      QN(("Q1", 0), ("Q2", 2)) => 1
      QN(("Q1", 1), ("Q2", 1)) => 1
      QN(("Q1", 0), ("Q2", -1)) => 1
      QN(("Q1", 1), ("Q2", 1)) => 1
      QN(("Q1", 2), ("Q2", 0)) => 1
    ]
  end
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

ITensors.state(::StateName"11", ::SiteType"SU(3)_conj") = [1.0, 0, 0, 0, 0, 0, 0, 0, 0]
ITensors.state(::StateName"12", ::SiteType"SU(3)_conj") = [0, 1.0, 0, 0, 0, 0, 0, 0, 0]
ITensors.state(::StateName"13", ::SiteType"SU(3)_conj") = [0, 0, 1.0, 0, 0, 0, 0, 0, 0]
ITensors.state(::StateName"21", ::SiteType"SU(3)_conj") = [0, 0, 0, 1.0, 0, 0, 0, 0, 0]
ITensors.state(::StateName"22", ::SiteType"SU(3)_conj") = [0, 0, 0, 0, 1.0, 0, 0, 0, 0]
ITensors.state(::StateName"23", ::SiteType"SU(3)_conj") = [0, 0, 0, 0, 0, 1.0, 0, 0, 0]
ITensors.state(::StateName"31", ::SiteType"SU(3)_conj") = [0, 0, 0, 0, 0, 0, 1.0, 0, 0]
ITensors.state(::StateName"32", ::SiteType"SU(3)_conj") = [0, 0, 0, 0, 0, 0, 0, 1.0, 0]
ITensors.state(::StateName"33", ::SiteType"SU(3)_conj") = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]

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

ITensors.op(::OpName"S1z",::SiteType"SU(3)_conj") =
kron(Sz, Id)

ITensors.op(::OpName"S2z",::SiteType"SU(3)_conj") =
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

    @eval begin
      ITensors.op(::OpName{$(QuoteNode(op_name))}, ::SiteType"SU(3)_conj") =
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

    @eval begin
      ITensors.op(::OpName{$(QuoteNode(op_name))}, ::SiteType"SU(3)_conj") =
      let matrix = zeros($(N_hilbert^2), $(N_hilbert^2))
        matrix1 = zeros($(N_hilbert), $(N_hilbert))
        matrix1[$i, $j] = 1.0
        matrix = kron(Id, matrix1)
        matrix
      end
    end
  end
end

function ITensors.op(::OpName"U(t)", ::SiteType"SU(3)", s1::Index, s2::Index; t, U)
  h = ITensor()

  for i in 1:3
    for j in 1:3
      h += op("S1_$(i)_$(j)", s1) * op("S1_$(j)_$(i)", s2)
      h += op("S2_$(i)_$(j)", s1) * op("S2_$(j)_$(i)", s2)

      for k in 1:3
        for l in 1:3
          h += replaceprime(U * op("S1_$(i)_$(j)", s1') * op("S1_$(j)_$(i)", s2') * op("S2_$(k)_$(l)", s1) * op("S2_$(l)_$(k)", s2), 2, 1)

        end
      end
    end
  end
  
  return exp(-im * t * h)
end

function ITensors.op(::OpName"U(t)", ::SiteType"SU(3)_conj", s1::Index, s2::Index; t, U)
  h = ITensor()

  for i in 1:3
    for j in 1:3
      h += op("S1_$(i)_$(j)", s1) * op("S1_$(j)_$(i)", s2)
      h += op("S2_$(i)_$(j)", s1) * op("S2_$(j)_$(i)", s2)

      for k in 1:3
        for l in 1:3
          h += replaceprime(U * op("S1_$(i)_$(j)", s1') * op("S1_$(j)_$(i)", s2') * op("S2_$(k)_$(l)", s1) * op("S2_$(l)_$(k)", s2), 2, 1)

        end
      end
    end
  end
  
  return exp(-im * t * h)
end

# Update block using swap SVDs (seems to lead to large errors)
# function updateblock(ψ, sites, i, Λs, W1, W2, cut, m)
#   W1 = replaceinds(W1, [sites[1],sites[1]',sites[3],sites[3]'], [sites[i],sites[i]',sites[i+2],sites[i+2]'])
#   W2 = replaceinds(W2, [sites[2],sites[2]',sites[4],sites[4]'], [sites[i+1],sites[i+1]',sites[i+3],sites[i+3]'])

#   # Swap middle two sites
#   Φ_bar = ψ[i+1] * ψ[i+2]
#   Φ = noprime(Λs[i] * Φ_bar)
#   # leftlink = dag(commonind(ψ[i], Φ))
#   U,S,V = svd(Φ, (sites[i+2], inds(Φ)[1]), cutoff=cut, maxdim=m, righttags="Link,l=$(i+1)", min_blockdim=1)
#   B3 = V
#   B2 = Φ_bar * dag(V)
#   # leftind = commonind(U,S)
#   # rightind = commonind(S,V)
#   # replaceind!(S, leftind, rightind')
#   Λs[i+1] = S

#   if (i == 1)
#     # Apply W1 to physical sites after swap
#     Φ_bar = ψ[i] * B2
#     Φ_bar = apply(W1, Φ_bar)
#     Φ = Φ_bar

#     U,S,V = svd(Φ, (sites[i]), cutoff=cut, maxdim=m, righttags="Link,l=$(i)", min_blockdim=1)
#     B2 = V
#     ψ[i] = Φ_bar * dag(V)
#     # leftind = commonind(U,S)
#     # rightind = commonind(S,V)
#     # replaceind!(S, leftind, rightind') # Priming the left side of Lambda, same index as right
#     Λs[i] = S

#     # Apply W2 to ancilla sites after swap
#     Φ_bar = B3 * ψ[i+3]
#     Φ_bar = apply(W2, Φ_bar)
#     Φ = noprime(Λs[i+1] * Φ_bar)

#     # leftlink = dag(commonind(B2, Φ))
#     U,S,V = svd(Φ, (sites[i+1], inds(Φ)[1]), cutoff=cut, maxdim=m, righttags="Link,l=$(i+2)", min_blockdim=1)
#     ψ[i+3] = V
#     B3 = Φ_bar * dag(V)
#     # leftind = commonind(U,S)
#     # rightind = commonind(S,V)
#     # replaceind!(S, leftind, rightind')
#     Λs[i+2] = S
#   else
#     # Apply W1 to physical sites after swap
#     Φ_bar = ψ[i] * B2
#     Φ_bar = apply(W1, Φ_bar)
#     Φ = noprime(Λs[i-1] * Φ_bar) # Priming the left side of Lambda, same index as right

#     # leftlink = dag(commonind(ψ[i-1], Φ))
#     U,S,V = svd(Φ, (sites[i], inds(Φ)[1]), cutoff=cut, maxdim=m, righttags="Link,l=$(i)", min_blockdim=1)
#     B2 = V
#     ψ[i] = Φ_bar * dag(V)
#     # leftind = commonind(U,S)
#     # rightind = commonind(S,V)
#     # replaceind!(S, leftind, rightind') # Priming the left side of Lambda, same index as right
#     Λs[i] = S

#     # Apply W2 to ancilla sites after swap
#     Φ_bar = B3 * ψ[i+3]
#     Φ_bar = apply(W2, Φ_bar)
#     Φ = noprime(Λs[i+1] * Φ_bar)

#     # leftlink = dag(commonind(B2, Φ))
#     U,S,V = svd(Φ, (sites[i+1], inds(Φ)[1]), cutoff=cut, maxdim=m, righttags="Link,l=$(i+2)", min_blockdim=1)
#     ψ[i+3] = V
#     B3 = Φ_bar * dag(V)
#     # leftind = commonind(U,S)
#     # rightind = commonind(S,V)
#     # replaceind!(S, leftind, rightind')
#     Λs[i+2] = S
#   end
  
#   # Swap middle two sites back
#   Φ_bar = B2 * B3
#   Φ = noprime(Λs[i] * Φ_bar)
#   # leftlink = dag(commonind(ψ[i], Φ))
#   U,S,V = svd(Φ, (sites[i+1], inds(Φ)[1]), cutoff=cut, maxdim=m, righttags="Link,l=$(i+1)", min_blockdim=1)
#   ψ[i+2] = V
#   ψ[i+1] = Φ_bar * dag(V)
#   # leftind = commonind(U,S)
#   # rightind = commonind(S,V)
#   # replaceind!(S, leftind, rightind')
#   Λs[i+1] = S
# end

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

# Update block by fully contracting four-site block and applying gates
# function updateblock(ψ, sites, i, Λs, W1, W2, cut, m)
#   Φ_bar = ψ[i] * ψ[i+1] * ψ[i+2] * ψ[i+3]
#   W1 = replaceinds(W1, [sites[1],sites[1]',sites[3],sites[3]'], [sites[i],sites[i]',sites[i+2],sites[i+2]'])
#   W2 = replaceinds(W2, [sites[2],sites[2]',sites[4],sites[4]'], [sites[i+1],sites[i+1]',sites[i+3],sites[i+3]'])
#   Φ_bar = apply(W1, Φ_bar)
#   Φ_bar = apply(W2, Φ_bar)

#   if (i == 1)
#     Φ = Φ_bar

#     U,S,V = svd(Φ, (sites[i], sites[i+1], sites[i+2]), cutoff=cut, maxdim=m, righttags="Link,l=$(i+2)")
#     ψ[i+3] = V
#     Φ = U*S
#     Φ_bar = Φ_bar * dag(V)
#     leftind = commonind(U,S)
#     rightind = commonind(S,V)
#     replaceind!(S, leftind, rightind')
#     Λs[i+2] = S

#     U,S,V = svd(Φ, (sites[i], sites[i+1]), cutoff=cut, maxdim=m, righttags="Link,l=$(i+1)")
#     ψ[i+2] = V
#     Φ = U*S
#     Φ_bar = Φ_bar * dag(V)
#     leftind = commonind(U,S)
#     rightind = commonind(S,V)
#     replaceind!(S, leftind, rightind')
#     Λs[i+1] = S

#     U,S,V = svd(Φ, (sites[i]), cutoff=cut, maxdim=m, righttags="Link,l=$(i)")
#     ψ[i+1] = V
#     ψ[i] = Φ_bar * dag(V)
#     leftind = commonind(U,S)
#     rightind = commonind(S,V)
#     replaceind!(S, leftind, rightind')
#     Λs[i] = S
#   else
#     Φ = noprime(Λs[i-1] * Φ_bar)

#     leftlink = dag(commonind(ψ[i-1], Φ))
#     U,S,V = svd(Φ, (sites[i], sites[i+1], sites[i+2], leftlink), cutoff=cut, maxdim=m, righttags="Link,l=$(i+2)")
#     ψ[i+3] = V
#     Φ = U*S
#     Φ_bar = Φ_bar * dag(V)
#     leftind = commonind(U,S)
#     rightind = commonind(S,V)
#     replaceind!(S, leftind, rightind')
#     Λs[i+2] = S

#     U,S,V = svd(Φ, (sites[i], sites[i+1], leftlink), cutoff=cut, maxdim=m, righttags="Link,l=$(i+1)")
#     ψ[i+2] = V
#     Φ = U*S
#     Φ_bar = Φ_bar * dag(V)
#     leftind = commonind(U,S)
#     rightind = commonind(S,V)
#     replaceind!(S, leftind, rightind')
#     Λs[i+1] = S

#     U,S,V = svd(Φ, (sites[i], leftlink), cutoff=cut, maxdim=m, righttags="Link,l=$(i)")
#     ψ[i+1] = V
#     ψ[i] = Φ_bar * dag(V)
#     leftind = commonind(U,S)
#     rightind = commonind(S,V)
#     replaceind!(S, leftind, rightind')
#     Λs[i] = S
#   end
# end

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

function create_gate_list(sites, δt, U)
  a1 = 0.095848502741203681182
  a2 = -0.078111158921637922695
  a3 = 0.5 - (a1 + a2)
  b1 = 0.42652466131587616168
  b2 = -0.12039526945509726545
  b3 = 1 - 2 * (b1 + b2)

  W1s = []
  W2s = []

  push!(W1s, op("U(t)", sites[1], sites[3], t = a1*δt, U=U))
  push!(W2s, op("U(t)", sites[2], sites[4], t = -a1*δt, U=U))
  push!(W1s, op("U(t)", sites[1], sites[3], t = b1*δt, U=U))
  push!(W2s, op("U(t)", sites[2], sites[4], t = -b1*δt, U=U))
  push!(W1s, op("U(t)", sites[1], sites[3], t = a2*δt, U=U))
  push!(W2s, op("U(t)", sites[2], sites[4], t = -a2*δt, U=U))
  push!(W1s, op("U(t)", sites[1], sites[3], t = b2*δt, U=U))
  push!(W2s, op("U(t)", sites[2], sites[4], t = -b2*δt, U=U))
  push!(W1s, op("U(t)", sites[1], sites[3], t = a3*δt, U=U))
  push!(W2s, op("U(t)", sites[2], sites[4], t = -a3*δt, U=U))
  push!(W1s, op("U(t)", sites[1], sites[3], t = b3*δt, U=U))
  push!(W2s, op("U(t)", sites[2], sites[4], t = -b3*δt, U=U))

  return W1s, W2s
end

# Gates for naive apply
function fourth_order_trotter_gates(L, sites, δt, U, real_evolution)
  a1 = 0.095848502741203681182
  a2 = -0.078111158921637922695
  a3 = 0.5 - (a1 + a2)
  b1 = 0.42652466131587616168
  b2 = -0.12039526945509726545
  b3 = 1 - 2 * (b1 + b2)

  A1 = ops([("U(t)", (2*n - 1, 2*n + 1), (t=a1*δt, U=U,)) for n in 1:2:(L - 1)], sites)
  B1 = ops([("U(t)", (2*n - 1, 2*n + 1), (t=b1*δt, U=U,)) for n in 2:2:(L - 2)], sites)
  A2 = ops([("U(t)", (2*n - 1, 2*n + 1), (t=a2*δt, U=U,)) for n in 1:2:(L - 1)], sites)
  B2 = ops([("U(t)", (2*n - 1, 2*n + 1), (t=b2*δt, U=U,)) for n in 2:2:(L - 2)], sites)
  A3 = ops([("U(t)", (2*n - 1, 2*n + 1), (t=a3*δt, U=U,)) for n in 1:2:(L - 1)], sites)
  B3 = ops([("U(t)", (2*n - 1, 2*n + 1), (t=b3*δt, U=U,)) for n in 2:2:(L - 2)], sites)

  if (real_evolution)
    # Apply disentangler exp(iHt) on ancilla sites
    aA1 = ops([("U(t)", (2*n, 2*n + 2), (t=-a1*δt, U=U,)) for n in 1:2:(L - 1)], sites)
    aB1 = ops([("U(t)", (2*n, 2*n + 2), (t=-b1*δt, U=U,)) for n in 2:2:(L - 2)], sites)
    aA2 = ops([("U(t)", (2*n, 2*n + 2), (t=-a2*δt, U=U,)) for n in 1:2:(L - 1)], sites)
    aB2 = ops([("U(t)", (2*n, 2*n + 2), (t=-b2*δt, U=U,)) for n in 2:2:(L - 2)], sites)
    aA3 = ops([("U(t)", (2*n, 2*n + 2), (t=-a3*δt, U=U,)) for n in 1:2:(L - 1)], sites)
    aB3 = ops([("U(t)", (2*n, 2*n + 2), (t=-b3*δt, U=U,)) for n in 2:2:(L - 2)], sites)

    A1 = vcat(A1,aA1)
    B1 = vcat(B1,aB1)
    A2 = vcat(A2,aA2)
    B2 = vcat(B2,aB2)
    A3 = vcat(A3,aA3)
    B3 = vcat(B3,aB3)
  end

  return vcat(A1,B1,A2,B2,A3,B3,A3,B2,A2,B1,A1)
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

  filename = "/pscratch/sd/k/kwang98/KPZ/tebd_su(3)_dw_L$(params.L)_chi$(params.maxdim)_beta$(params.β_max)_dt$(params.δt)_U$(params.U)_mu$(params.μ)_conserve_threesite_conj.h5"
  # filename = "tebd_su(3)_dw_MPI_L$(params.L)_chi$(params.maxdim)_beta$(params.β_max)_dt$(params.δt)_U$(params.U)_mu$(params.μ).h5"

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
    orthogonalize!(ψ, 1)
    Λs = find_lambdas(ψ)
    W1s, W2s = create_gate_list(sites, params.δt, params.U)
    # real_gates = fourth_order_trotter_gates(params.L, sites, params.δt, params.U, true)
  else
    sites = siteinds(n->isodd(n) ? "SU(3)" : "SU(3)_conj", 2 * params.L; conserve_qns=true)
    # sites = siteinds("SU(3)", 2 * params.L; conserve_qns=true)
    W1s, W2s = create_gate_list(sites, params.δt, params.U)
    # real_gates = fourth_order_trotter_gates(params.L, sites, params.δt, params.U, true)
  
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

    @time fourth_order_trotter_sweep(ψ, sites, Λs, W1s, W2s, params.cutoff, params.maxdim)
    # @time ψ = apply(real_gates, ψ; params.cutoff, params.maxdim)
    GC.gc()

    Z1 = expect(ψ, "S1z"; sites=1:2:(2*params.L-1))
    Z2 = expect(ψ, "S2z"; sites=1:2:(2*params.L-1))
    S = entropy_von_neumann(ITensors.cpu(ψ), 2*params.L) # Von neumann entropy at half-cut between ancilla and physical (initially unentangled)
    # @time M_moment = moments(L, ψ, sites, cutoff, maxdim)

    println("Time = $t")
    println("Max bond dimension = $(maxlinkdim(ψ))")
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
    parse(Float64, ARGS[6])  # μ
)

# L = parse(Int64, ARGS[1])
# maxdim = parse(Int64, ARGS[2])
# β_max = parse(Float64, ARGS[3])
# δt = parse(Float64, ARGS[4])
# U = parse(Float64, ARGS[5])
# μ = parse(Float64, ARGS[6])

# main(L=L, maxdim=maxdim, β_max=β_max, δt=δt, U=U, μ=μ)
main(params)