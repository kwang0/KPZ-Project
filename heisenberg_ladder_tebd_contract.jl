using MKL
using LinearAlgebra
using ITensors
using Printf
using PyPlot
using HDF5

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

function ITensors.op(::OpName"expiSS", ::SiteType"S=1/2", s1::Index, s2::Index; t)
  h =
    1 / 2 * op("S+", s1) * op("S-", s2) +
    1 / 2 * op("S-", s1) * op("S+", s2) +
    op("Sz", s1) * op("Sz", s2)
  return exp(-im * t * h)
end

# Forward and backward trotter sweep
function trotter_sweep(ψ, L, sites, t, real_evolution, cut, m)
  # Initial edge case
  three_site = ψ[1]*ψ[2]*ψ[3]
  three_site = apply(op("expiSS", sites[1], sites[3], t=t), three_site)
  U,S,V = svd(three_site, (sites[1]), cutoff=cut, maxdim=m, lefttags="Link,l=1")
  ψ[1] = U
  two_site = S*V

  # Forward sweep
  for i in 2:1:(2*L-2)
    three_site = two_site * ψ[i+2]
    if (i%2 == 1)
      three_site = apply(op("expiSS", sites[i], sites[i+2], t=t), three_site)
    elseif (real_evolution)
      three_site = apply(op("expiSS", sites[i], sites[i+2], t=-t), three_site)
    end
    leftlink = dag(commonind(ψ[i-1], three_site))
    U,S,V = svd(three_site, (sites[i], leftlink), cutoff=cut, maxdim=m, lefttags="Link,l=$(i)")
    ψ[i] = U
    two_site = S*V
  end

  # Backward sweep
  for i in (2*L-2):-1:2
    three_site = two_site * ψ[i]
    if (i%2 == 1)
      three_site = apply(op("expiSS", sites[i], sites[i+2], t=t), three_site)
    elseif (real_evolution)
      three_site = apply(op("expiSS", sites[i], sites[i+2], t=-t), three_site)
    end
    leftlink = dag(commonind(ψ[i-1], three_site))
    U,S,V = svd(three_site, (sites[i], sites[i+1], leftlink), cutoff=cut, maxdim=m, lefttags="Link,l=$(i+1)")
    two_site = U
    ψ[i+2] = S*V
  end

  # Final edge case
  three_site = two_site * ψ[1]
  three_site = apply(op("expiSS", sites[1], sites[3], t=t), three_site)
  U,S,V = svd(three_site, (sites[1], sites[2]), cutoff=cut, maxdim=m, lefttags="Link,l=2")
  two_site = U
  ψ[3] = S*V
  U,S,V = svd(two_site, (sites[1]), cutoff=cut, maxdim=m, lefttags="Link,l=1")
  ψ[1] = U
  ψ[2] = S*V

  return ψ
end

function main(; L=128, cutoff=1E-16, δτ=0.05, beta_max=3.0, δt=0.1, ttotal=100, maxdim=32)
  s = siteinds("S=1/2", 2 * L; conserve_qns=true)

  # Initial state is infinite-temperature mixed state (purification)
  ψ = inf_temp_mps(s)

  # Parameter for fourth order trotterization
  α = 1.0 / (2 - 2^(1/3))

  # Cool down to inverse temperature 
  for β in δτ:δτ:beta_max/2
    @printf("β = %.2f\n", 2*β)
    flush(stdout)
    ψ = trotter_sweep(ψ, L, s, (α/2) * δτ, false, cutoff, maxdim)
    ψ = trotter_sweep(ψ, L, s, ((1-2*α)/2) * δτ, false, cutoff, maxdim)
    ψ = trotter_sweep(ψ, L, s, (α/2) * δτ, false, cutoff, maxdim)
    normalize!(ψ)
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
    F = h5open("data_jl/tebd_L$(L)_chi$(maxdim)_beta$(beta_max)_dt$(δt)_contract.h5","w")
    F["times"] = times
    F["corrs"] = corrs
    close(F)

    t≈ttotal && break

    ψ = trotter_sweep(ψ, L, s, (α/2) * δt, true, cutoff, maxdim)
    ψ = trotter_sweep(ψ, L, s, ((1-2*α)/2) * δt, true, cutoff, maxdim)
    ψ = trotter_sweep(ψ, L, s, (α/2) * δt, true, cutoff, maxdim)
    normalize!(ψ)
    ψ2 = trotter_sweep(ψ2, L, s, (α/2) * δt, true, cutoff, maxdim)
    ψ2 = trotter_sweep(ψ2, L, s, ((1-2*α)/2) * δt, true, cutoff, maxdim)
    ψ2 = trotter_sweep(ψ2, L, s, (α/2) * δt, true, cutoff, maxdim)
    normalize!(ψ2)

    println("Max bond dimension is $(maxlinkdim(ψ2))")
  end

  # plt.loglog(times, abs.(corrs))
  # plt.xlabel("t")
  # plt.ylabel("|C(T,x=0,t)|")
  # plt.show()

  return times, corrs
end

ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(40)
# ITensors.enable_threaded_blocksparse(true)

L = parse(Int64, ARGS[1])
maxdim = parse(Int64, ARGS[2])
beta_max = parse(Float64, ARGS[3])
δt = parse(Float64, ARGS[4])

main(L=L, maxdim=maxdim, beta_max=beta_max, δt=δt)