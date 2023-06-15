using MKL
using ITensorTDVP
using ITensorGPU
using ITensors
using CUDA
using Printf
using PyPlot
using HDF5
using LinearAlgebra
include("basis_extend.jl")

function heisenberg(L, J2, real_evolution)
  os = OpSum()

  # Adding J1 = 1 terms in ladder
  for j in 1:2:(4*L - 5)
    os += "Sz", j, "Sz", j + 4
    os += 0.5, "S+", j, "S-", j + 4
    os += 0.5, "S-", j, "S+", j + 4

    if (real_evolution)
      # Apply disentangler exp(iHt) on ancilla sites
      os += -1, "Sz", j + 1, "Sz", j + 5
      os += -0.5, "S+", j + 1, "S-", j + 5
      os += -0.5, "S-", j + 1, "S+", j + 5
    end
  end

  # Adding J2 rung terms in ladder
  for j in 1:4:(4*L - 3)
    os += J2, "Sz", j, "Sz", j + 2
    os += 0.5*J2, "S+", j, "S-", j + 2
    os += 0.5*J2, "S-", j, "S+", j + 2

    if (real_evolution)
      # Apply disentangler exp(iHt) on ancilla sites
      os += -1*J2, "Sz", j + 1, "Sz", j + 3
      os += -0.5*J2, "S+", j + 1, "S-", j + 3
      os += -0.5*J2, "S-", j + 1, "S+", j + 3
    end
  end

  return os
end

# function heisenberg(L)
#   os = OpSum()
#   for j in 1:(L - 1)
#     os += "Sz", 2*j - 1, "Sz", 2*j + 1
#     os += 0.5, "S+", 2*j - 1, "S-", 2*j + 1
#     os += 0.5, "S-", 2*j - 1, "S+", 2*j + 1
#   end
#   return os
# end

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

function main(; L=128, cutoff=1e-16, δτ=0.05, β_max=3.0, δt=0.1, ttotal=100, maxdim=32, J2=0)
  sites = siteinds("S=1/2", 4 * L; conserve_qns=false)
  H_imag = cuMPO(MPO(heisenberg(L, J2, false), sites))
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

  c = div(L, 2) # center site
  Sz_center = op("Sz",sites[4*c-3])
  orthogonalize!(ψ, 4*c-3)
  ψ2 = apply(2 * Sz_center, ψ; cutoff, maxdim)
  # normalize!(ψ2)

  times = Float64[]
  corrs = ComplexF64[]
  ψ_norms = Float64[]
  ψ2_norms = Float64[]
  converted_to_gpu = false
  for t in 0.0:δt:ttotal
    orthogonalize!(ψ, 4*c-3)
    ψ3 = apply(2 * Sz_center, ψ2; cutoff, maxdim)
    # normalize!(ψ3)
    corr = inner(ψ, ψ3)
    println("$t $corr")
    flush(stdout)
    push!(times, t)
    push!(corrs, corr)
    push!(ψ_norms, norm(ψ))
    push!(ψ2_norms, norm(ψ2))

    # Writing to data file
    F = h5open("data_jl/tdvp_gpu_L$(L)_chi$(maxdim)_beta$(β_max)_dt$(δt)_Jprime$(J2)_diskwrite.h5","w")
    F["times"] = times
    F["corrs"] = corrs
    F["psi_norms"] = ψ_norms
    F["psi2_norms"] = ψ2_norms
    close(F)

    t≈ttotal && break

    # ψ = basis_extend(ψ, H_real; cutoff, extension_krylovdim=2)
    if (maxlinkdim(ψ2) < 0)
      ψ2 = basis_extend(ψ2, H_real; cutoff, extension_krylovdim=2)
    elseif (!converted_to_gpu)
      H_real = cuMPO(H_real)
      ψ = cuMPS(ψ)
      ψ2 = cuMPS(ψ2)
      Sz_center = cuITensor(Sz_center)
      converted_to_gpu = true
    end

    ψ = tdvp(H_real, -im * δt, ψ;
      nsweeps=1,
      reverse_step=true,
      normalize=false,
      maxdim=maxdim,
      cutoff=cutoff,
      outputlevel=1
    )
    ψ2 = tdvp(H_real, -im * δt, ψ2;
      nsweeps=1,
      reverse_step=true,
      normalize=false,
      maxdim=maxdim,
      cutoff=cutoff,
      outputlevel=1,
      write_when_maxdim_exceeds=10
    )
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
β_max = parse(Float64, ARGS[3])
δt = parse(Float64, ARGS[4])
J2 = parse(Float64, ARGS[5])

main(L=L, maxdim=maxdim, β_max=β_max, δt=δt, J2=J2)