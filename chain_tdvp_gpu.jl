using MKL
using ITensors
using ITensorMPS
using CUDA
using Printf
using PyPlot
using HDF5
using LinearAlgebra
using TickTock

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
    ψ = MPS(sites, state)
    for j = 1:2:num_sites-1
      s1 = sites[j]
      s2 = sites[j+1]
          
      if(j == 1)
        rightlink = commonind(ψ[j+1],ψ[j+2])
        A = ITensor(ComplexF64, s1, s2, rightlink)

        A[s1=>1, s2=>2, rightlink => 1] = 1/2
        A[s1=>2, s2=>1, rightlink => 1] = -1/2

        U,S,V = svd(A, (s1), cutoff=1e-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V

      elseif (j == num_sites-1)
        leftlink = dag(commonind(ψ[j-1], ψ[j]))
        A = ITensor(ComplexF64, s1, s2, leftlink)

        A[s1=>1, s2=>2, leftlink => 1] = 1/2
        A[s1=>2, s2=>1, leftlink => 1] = -1/2

        U,S,V = svd(A, (s1, leftlink), cutoff=1e-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V
        
      else
        rightlink = commonind(ψ[j+1], ψ[j+2])
        leftlink = dag(commonind(ψ[j-1], ψ[j]))
    
        A = ITensor(ComplexF64, s1, s2, rightlink, leftlink)

        A[s1=>1, s2=>2, rightlink=>1, leftlink => 1] = 1/2
        A[s1=>2, s2=>1, rightlink=>1, leftlink => 1] = -1/2

        U,S,V = svd(A, (s1, leftlink), cutoff=1e-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V
      end
    end

    return ψ
  end
end

function heisenberg(L, real_evolution)
  os = OpSum()

  # Adding J1 = 1 terms
  for j in 1:2:(2*L - 3)
    os += 1, "Sz", j, "Sz", j + 2
    os += 0.5, "S+", j, "S-", j + 2
    os += 0.5, "S-", j, "S+", j + 2

    if (real_evolution)
      # Apply disentangler exp(iHt) on ancilla sites
      os += -1, "Sz", j + 1, "Sz", j + 3
      os += -0.5, "S+", j + 1, "S-", j + 3
      os += -0.5, "S-", j + 1, "S+", j + 3
    end
  end

  return os
end

# Adding "Zeeman terms" to produce domain wall density matrix
function H_dw(L)
  os = OpSum()

  for j in 1:2:(L - 1)
    os += 1, "Sz", j
  end

  for j in (L+1):2:(2*L - 1)
    os -= 1, "Sz", j
  end
  
  return os
end

function main(; L=128, cutoff=1e-12, δt=0.1, ttotal=200, maxdim=32, μ=0.001)
  tick()

  c = div(L,2) + 1 # center site

  filename = "/pscratch/sd/k/kwang98/KPZ/production/chain_L$(L)_chi$(maxdim)_mu$(μ)_1e12.h5"

  if (isfile(filename))
    F = h5open(filename,"r")
    times = read(F, "times")
    Zs = read(F, "Zs")
    Ss = read(F, "Ss")
    ψ = cu(read(F, "psi", MPS))
    start_time = last(times) + δt
    close(F)

    sites = siteinds(ψ)
    H_real = cu(MPO(heisenberg(L, true), sites))
  else
    sites = siteinds("S=1/2", 2 * L; conserve_qns=false)
    H_real = cu(MPO(heisenberg(L, true), sites))
  
    # Initial state is infinite-temperature mixed state, odd = physical, even = ancilla
    ψ = cu(inf_temp_mps(sites))
  
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

    times = Float64[]
    Zs = []
    Ss = []
    start_time = δt
  end

  for t in start_time:δt:ttotal
    # Stop simulations before HPC limit to ensure no corruption of data writing
    if peektimer() > (23.5 * 60 * 60)
      break
    end

    ψ = tdvp(H_real, -im * δt, ψ;
      updater_backend="applyexp",
      nsweeps=1,
      reverse_step=true,
      normalize=false,
      maxdim=maxdim,
      cutoff=cutoff,
      outputlevel=1,
      nsite=2
    )
    GC.gc()

    Z = expect(ψ, "Sz"; sites=1:2:(2*L-1))
    S = entropy_von_neumann(ITensors.cpu(ψ), L) # Von neumann entropy at half-cut between ancilla and physical (initially unentangled)

    println("Time = $t")
    flush(stdout)
    push!(times, t)
    t == δt ? Zs = Z : Zs = hcat(Zs, Z)
    t == δt ? Ss = S : Ss = hcat(Ss, S)

    # Writing to data file
    F = h5open(filename,"w")
    F["times"] = times
    F["Zs"] = Zs
    F["Ss"] = Ss
    F["corrs"] = (Zs[c-1,:] .- Zs[c,:]) ./ (2 * μ)
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
μ = parse(Float64, ARGS[3])

main(L=L, maxdim=maxdim, μ=μ)