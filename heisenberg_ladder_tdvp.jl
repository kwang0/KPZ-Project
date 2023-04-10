using MKL
using ITensors
using ITensorTDVP
using Printf
using PyPlot
using HDF5
using LinearAlgebra

function heisenberg(L, J2)
  os = OpSum()
  for j in 1:2:(4*L - 3)
    # Adding J1 = 1 terms in ladder
    os += "Sz", j, "Sz", j + 4
    os += 0.5, "S+", j, "S-", j + 4
    os += 0.5, "S-", j, "S+", j + 4

    # Adding J2 rung terms in ladder
    if (j % 4 == 1)
      os += J2, "Sz", j, "Sz", j + 2
      os += 0.5*J2, "S+", j, "S-", j + 2
      os += 0.5*J2, "S-", j, "S+", j + 2
    end
  end
  return os
end

function inf_temp_mps(sites)
  num_sites = length(sites)
  if (num_sites % 2 != 0)
    throw(DomainError(num_sites,"Expects even number of sites for ancilla-physical singlets."))
  else
    ψ = MPS(sites)
    for j = 1:2:num_sites-1
      s1 = sites[j]
      s2 = sites[j+1]
          
      if(j == 1)
        rightlink = commonind(ψ[j+1],ψ[j+2])
        A = ITensor(ComplexF64, s1, s2, rightlink)

        A[s1=>1, s2=>2, rightlink => 1] = 1/sqrt(2)
        A[s1=>2, s2=>1, rightlink => 1] = -1/sqrt(2)

        U,S,V = svd(A, (s1), cutoff=1e-15)
        ψ[j] = replacetags(U, "u", "l=$(j)")
        ψ[j+1] = replacetags(S*V, "u", "l=$(j)")

      elseif (j == num_sites-1)
        leftlink = dag(commonind(ψ[j-1], ψ[j]))
        A = ITensor(ComplexF64, s1, s2, leftlink)

        A[s1 => 1,s2 => 2, leftlink => 1] = 1/sqrt(2)
        A[s1 => 2,s2 => 1, leftlink => 1] = -1/sqrt(2)

        U,S,V = svd(A, (s1, leftlink), cutoff=1e-15)
        ψ[j] = replacetags(U, "u", "l=$(j)")
        ψ[j+1] = replacetags(S*V, "u", "l=$(j)")
        
      else
        rightlink = commonind(ψ[j+1], ψ[j+2])
        leftlink = dag(commonind(ψ[j-1], ψ[j]))
    
        A = ITensor(ComplexF64, s1, s2, rightlink, leftlink)

        A[s1 => 1,s2 => 2, rightlink=>1, leftlink =>1] = 1/sqrt(2)
        A[s1 => 2,s2 => 1, rightlink=>1, leftlink =>1] = -1/sqrt(2)

        U,S,V = svd(A, (s1, leftlink), cutoff=1e-15)
        ψ[j] = replacetags(U, "u", "l=$(j)")
        ψ[j+1] = replacetags(S*V, "u", "l=$(j)")
      end
    end

    return ψ
  end
end

function main(; L=128, cutoff=1e-10, δτ=0.05, β_max=3.0, δt=0.1, ttotal=100, maxdim=32, J2=0)
  s = siteinds("S=1/2", 4 * L; conserve_qns=true)
  H = MPO(heisenberg(L, J2), s)

  # Initial state is infinite-temperature mixed state, odd = physical, even = ancilla
  ψ = inf_temp_mps(s)

  # Cool down to inverse temperature 
  for β in 0:δτ:β_max/2
    @printf("β = %.2f\n", 2*β)
    flush(stdout)
    ψ = tdvp(H, -δτ, ψ;
      nsweeps=1,
      reverse_step=true,
      normalize=true,
      maxdim=maxdim,
      cutoff=cutoff,
      outputlevel=1
    )
  end

  c = div(L, 2) # center site
  Sz_center = op("Sz",s[2*c-1])
  ψ2 = apply(Sz_center, ψ; cutoff, maxdim)
  normalize!(ψ2)

  times = Float64[]
  corrs = ComplexF64[]
  for t in 0.0:δt:ttotal
    ψ3 = apply(Sz_center, ψ2; cutoff, maxdim)
    normalize!(ψ3)
    corr = inner(ψ, ψ3)
    println("$t $corr")
    flush(stdout)
    push!(times, t)
    push!(corrs, corr)

    # Writing to data file
    F = h5open("data_jl/tdvp_L$(L)_chi$(maxdim)_beta$(β_max)_dt$(δt)_J2$(J2).h5","w")
    F["times"] = times
    F["corrs"] = corrs
    close(F)

    t≈ttotal && break

    ψ = tdvp(H, -im * δt, ψ;
      nsweeps=1,
      reverse_step=true,
      normalize=true,
      maxdim=maxdim,
      cutoff=cutoff,
      outputlevel=1
    )
    ψ2 = tdvp(H, -im * δt, ψ2;
      nsweeps=1,
      reverse_step=true,
      normalize=true,
      maxdim=maxdim,
      cutoff=cutoff,
      outputlevel=1
    )
  end

  plt.loglog(times, abs.(corrs))
  plt.xlabel("t")
  plt.ylabel("|C(T,x=0,t)|")
  plt.show()

  return times, corrs
end

ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(40)
# ITensors.enable_threaded_blocksparse(true)

L = parse(Int64, ARGS[1])
maxdim = parse(Int64, ARGS[2])
β_max = parse(Float64, ARGS[3])
δt = parse(Float64, ARGS[4])
J2 = parse(Float64, ARGS[5])

main(L=L, maxdim=maxdim, β_max=β_max, δt=δt, J2=J2)