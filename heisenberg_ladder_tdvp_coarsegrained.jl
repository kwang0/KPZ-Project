using MKL
using ITensors
using ITensorTDVP
using Printf
using PyPlot
using HDF5
using LinearAlgebra
include("basis_extend.jl")

mutable struct SizeObserver <: AbstractObserver
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

function heisenberg(L, J2, real_evolution)
  os = OpSum()

  # Adding J1 = 1 terms in ladder
  for j in 1:2:(2*L - 3)
    os += "S1z", j, "S1z", j + 2
    os += 0.5, "S1+", j, "S1-", j + 2
    os += 0.5, "S1-", j, "S1+", j + 2

    os += "S2z", j, "S2z", j + 2
    os += 0.5, "S2+", j, "S2-", j + 2
    os += 0.5, "S2-", j, "S2+", j + 2

    if (real_evolution)
      # Apply disentangler exp(iHt) on ancilla sites
      os += -1, "S1z", j + 1, "S1z", j + 3
      os += -0.5, "S1+", j + 1, "S1-", j + 3
      os += -0.5, "S1-", j + 1, "S1+", j + 3

      os += -1, "S2z", j + 1, "S2z", j + 3
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

  return os
end

function main(; L=128, cutoff=1e-10, δτ=0.05, β_max=3.0, δt=0.1, ttotal=100, maxdim=32, J2=0)
  sites = siteinds("S=3/2", 2 * L; conserve_qns=true)
  H_imag = MPO(heisenberg(L, J2, false), sites)
  H_real = MPO(heisenberg(L, J2, true), sites)

  # Initial state is infinite-temperature mixed state, odd = physical, even = ancilla
  ψ = inf_temp_mps(sites)
  # ψ = basis_extend(ψ, H_real; cutoff, extension_krylovdim=2)

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
      outputlevel=1
    )
  end

  c = L - 1 # center site
  Sz_center = op("S1z",sites[c])
  orthogonalize!(ψ, c)
  ψ2 = apply(2 * Sz_center, ψ; cutoff, maxdim)
  # normalize!(ψ2)

  # filename = "/global/scratch/users/kwang98/KPZ/tdvp_coarsegrained_L$(L)_chi$(maxdim)_beta$(β_max)_dt$(δt)_Jprime$(J2).h5"
  filename = "/pscratch/sd/k/kwang98/KPZ/tdvp_coarsegrained_L$(L)_chi$(maxdim)_beta$(β_max)_dt$(δt)_Jprime$(J2).h5"
  # filename = "tdvp_coarsegrained_L$(L)_chi$(maxdim)_beta$(β_max)_dt$(δt)_Jprime$(J2).h5"

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
    H_real = MPO(heisenberg(L, J2, true), sites)
  else
    times = Float64[]
    corrs = []
    ψ_norms = Float64[]
    ψ2_norms = Float64[]
    start_time = 0.0
  end

  obs = SizeObserver()
  for t in start_time:δt:ttotal
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

    # ψ = basis_extend(ψ, H_real; cutoff, extension_krylovdim=2)
    # if (maxlinkdim(ψ2) < 100)
    # @time ψ2 = basis_extend(ψ2, H_real; cutoff, extension_krylovdim=2)
    # end

    ψ = tdvp(H_real, -im * δt, ψ;
      nsweeps=1,
      reverse_step=true,
      normalize=false,
      maxdim=maxdim,
      cutoff=cutoff,
      outputlevel=1,
      (observer!)=obs
    )
    GC.gc()
    ψ2 = tdvp(H_real, -im * δt, ψ2;
      nsweeps=1,
      reverse_step=true,
      normalize=false,
      maxdim=maxdim,
      cutoff=cutoff,
      outputlevel=1,
      (observer!)=obs
    )
    GC.gc()
  end

  # plt.loglog(times, abs.(corrs[L-1,:]))
  # plt.xlabel("t")
  # plt.ylabel("|C(T,x=0,t)|")
  # plt.show()

  return times, corrs
end

ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(256)
# ITensors.enable_threaded_blocksparse(true)

L = parse(Int64, ARGS[1])
maxdim = parse(Int64, ARGS[2])
β_max = parse(Float64, ARGS[3])
δt = parse(Float64, ARGS[4])
J2 = parse(Float64, ARGS[5])

main(L=L, maxdim=maxdim, β_max=β_max, δt=δt, J2=J2)