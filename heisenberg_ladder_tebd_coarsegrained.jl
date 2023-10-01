using MKL
using LinearAlgebra
using ITensors
using Printf
using PyPlot
using HDF5
using TickTock

function inf_temp_mps(sites)
  num_sites = length(sites)
  if (num_sites % 2 != 0)
    throw(DomainError(num_sites,"Expects even number of sites for ancilla-physical singlets."))
  else
    # state = [isodd(n) ? "Up" : "Dn" for n=1:num_sites]
    ψ = randomMPS(sites)
    for j = 1:2:num_sites-1
      s1 = sites[j]
      s2 = sites[j+1]
          
      if(j == 1)
        rightlink = commonind(ψ[j+1],ψ[j+2])
        A = ITensor(ComplexF64, s1, s2, rightlink)

        A[s1=>1, s2=>1, rightlink => 1] = 1/2
        A[s1=>4, s2=>4, rightlink => 1] = 1/2
        A[s1=>2, s2=>2, rightlink => 1] = 1/2
        A[s1=>3, s2=>3, rightlink => 1] = 1/2

        U,S,V = svd(A, (s1), cutoff=1e-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V

      elseif (j == num_sites-1)
        leftlink = dag(commonind(ψ[j-1], ψ[j]))
        A = ITensor(ComplexF64, s1, s2, leftlink)

        A[s1=>1, s2=>1, leftlink => 1] = 1/2
        A[s1=>4, s2=>4, leftlink => 1] = 1/2
        A[s1=>2, s2=>2, leftlink => 1] = 1/2
        A[s1=>3, s2=>3, leftlink => 1] = 1/2

        U,S,V = svd(A, (s1, leftlink), cutoff=1e-16, lefttags="Link,l=$(j)")
        ψ[j] = U
        ψ[j+1] = S*V
        
      else
        rightlink = commonind(ψ[j+1], ψ[j+2])
        leftlink = dag(commonind(ψ[j-1], ψ[j]))
    
        A = ITensor(ComplexF64, s1, s2, rightlink, leftlink)

        A[s1=>1, s2=>1, rightlink=>1, leftlink => 1] = 1/2
        A[s1=>4, s2=>4, rightlink=>1, leftlink => 1] = 1/2
        A[s1=>2, s2=>2, rightlink=>1, leftlink => 1] = 1/2
        A[s1=>3, s2=>3, rightlink=>1, leftlink => 1] = 1/2

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

ITensors.op(::OpName"S1z",::SiteType"S=3/2") =
  [+1/2   0    0    0
     0  +1/2   0    0 
     0    0  -1/2   0
     0    0    0  -1/2]
     
ITensors.op(::OpName"S2z",::SiteType"S=3/2") =
  [+1/2   0    0    0
   0  -1/2   0    0 
   0    0  -1/2   0
   0    0    0  +1/2]

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
    0     1/2   1/4   0
    0     0     0    -1/4]
ITensors.op(::OpName"Id",::SiteType"S=3/2") =
  [1   0  0   0
   0   1  0   0
   0   0  1   0
   0   0  0  1]

function ITensors.op(::OpName"expiSS", ::SiteType"S=3/2", s1::Index, s2::Index; t, J1, J2)
  h1 =
    1 / 2 * op("S1+", s1) * op("S1-", s2) +
    1 / 2 * op("S1-", s1) * op("S1+", s2) +
    op("S1z", s1) * op("S1z", s2)
  h2 =
    1 / 2 * op("S2+", s1) * op("S2-", s2) +
    1 / 2 * op("S2-", s1) * op("S2+", s2) +
    op("S2z", s1) * op("S2z", s2)
  rung = op("rung", s1) * op("Id", s2)
  
  return exp(-im * t * (J1 * (h1 + h2) + J2 * rung))
end

# function fourth_order_trotter_gates(L, sites, δt, real_evolution)
#   α = 1 / (4 - 4^(1/3))

#   A1 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=α*δt/2,)) for n in 1:2:(L - 1)], sites)
#   B1 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=α*δt,)) for n in 2:2:(L - 2)], sites)
#   A2 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=α*δt,)) for n in 1:2:(L - 1)], sites)
#   A3 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=(1-3*α)*δt/2,)) for n in 1:2:(L - 1)], sites)
#   B2 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=(1-4*α)*δt,)) for n in 2:2:(L - 2)], sites)

#   if (real_evolution)
#     # Apply disentangler exp(iHt) on ancilla sites
#     aA1 = ops([("expiSS", (2*n, 2*n + 2), (t=-α*δt/2,)) for n in 1:2:(L - 1)], sites)
#     aB1 = ops([("expiSS", (2*n, 2*n + 2), (t=-α*δt,)) for n in 2:2:(L - 2)], sites)
#     aA2 = ops([("expiSS", (2*n, 2*n + 2), (t=-α*δt,)) for n in 1:2:(L - 1)], sites)
#     aA3 = ops([("expiSS", (2*n, 2*n + 2), (t=-(1-3*α)*δt/2,)) for n in 1:2:(L - 1)], sites)
#     aB2 = ops([("expiSS", (2*n, 2*n + 2), (t=-(1-4*α)*δt,)) for n in 2:2:(L - 2)], sites)

#     A1 = vcat(A1,aA1)
#     B1 = vcat(B1,aB1)
#     A2 = vcat(A2,aA2)
#     A3 = vcat(A3,aA3)
#     B2 = vcat(B2,aB2)
#   end

#   return vcat(A1,B1,A2,B1,A3,B2,A3,B1,A2,B1,A1)
# end

function fourth_order_trotter_gates(L, sites, δt, J1, J2, real_evolution)
  a1 = 0.095848502741203681182
  a2 = -0.078111158921637922695
  a3 = 0.5 - (a1 + a2)
  b1 = 0.42652466131587616168
  b2 = -0.12039526945509726545
  b3 = 1 - 2 * (b1 + b2)

  A1 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=a1*δt, J1=J1, J2=J2,)) for n in 1:2:(L - 1)], sites)
  B1 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=b1*δt, J1=J1, J2=J2,)) for n in 2:2:(L - 2)], sites)
  A2 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=a2*δt, J1=J1, J2=J2,)) for n in 1:2:(L - 1)], sites)
  B2 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=b2*δt, J1=J1, J2=J2,)) for n in 2:2:(L - 2)], sites)
  A3 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=a3*δt, J1=J1, J2=J2,)) for n in 1:2:(L - 1)], sites)
  B3 = ops([("expiSS", (2*n - 1, 2*n + 1), (t=b3*δt, J1=J1, J2=J2,)) for n in 2:2:(L - 2)], sites)

  if (real_evolution)
    # Apply disentangler exp(iHt) on ancilla sites
    aA1 = ops([("expiSS", (2*n, 2*n + 2), (t=-a1*δt, J1=J1, J2=J2,)) for n in 1:2:(L - 1)], sites)
    aB1 = ops([("expiSS", (2*n, 2*n + 2), (t=-b1*δt, J1=J1, J2=J2,)) for n in 2:2:(L - 2)], sites)
    aA2 = ops([("expiSS", (2*n, 2*n + 2), (t=-a2*δt, J1=J1, J2=J2,)) for n in 1:2:(L - 1)], sites)
    aB2 = ops([("expiSS", (2*n, 2*n + 2), (t=-b2*δt, J1=J1, J2=J2,)) for n in 2:2:(L - 2)], sites)
    aA3 = ops([("expiSS", (2*n, 2*n + 2), (t=-a3*δt, J1=J1, J2=J2,)) for n in 1:2:(L - 1)], sites)
    aB3 = ops([("expiSS", (2*n, 2*n + 2), (t=-b3*δt, J1=J1, J2=J2,)) for n in 2:2:(L - 2)], sites)

    A1 = vcat(A1,aA1)
    B1 = vcat(B1,aB1)
    A2 = vcat(A2,aA2)
    B2 = vcat(B2,aB2)
    A3 = vcat(A3,aA3)
    B3 = vcat(B3,aB3)
  end

  return vcat(A1,B1,A2,B2,A3,B3,A3,B2,A2,B1,A1)
end

function main(; L=128, cutoff=1E-16, δτ=0.05, β_max=3.0, δt=0.1, ttotal=100, maxdim=32, J1=1.0, J2=0.0)
  # Make purification gates
  # im_gates = fourth_order_trotter_gates(L, s, -im * δτ, J2, false)

  # Cool down to inverse temperature 
  # for β in δτ:δτ:β_max/2
  #   @printf("β = %.2f\n", 2*β)
  #   flush(stdout)
  #   ψ = apply(im_gates, ψ; cutoff, maxdim)
  #   normalize!(ψ)
  # end

  c = L - 1 # center site

  filename = "/pscratch/sd/k/kwang98/KPZ/tebd_coarsegrained_L$(L)_chi$(maxdim)_beta$(β_max)_dt$(δt)_Jprime$(J2).h5"
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
    # Make real-time evolution gates
    real_gates = fourth_order_trotter_gates(L, sites, δt, J1, J2, true)
    GC.gc()
  else
    sites = siteinds("S=3/2", 2 * L; conserve_qns=false)

    # Initial state is infinite-temperature mixed state (purification)
    ψ = inf_temp_mps(sites)
    real_gates = fourth_order_trotter_gates(L, sites, δt, J1, J2, true)
    GC.gc()
    
    Sz_center = op("S1z",sites[c])
    orthogonalize!(ψ, c)
    ψ2 = apply(2 * Sz_center, ψ; cutoff, maxdim)
    # normalize!(ψ2)
  
    times = Float64[]
    corrs = []
    ψ_norms = Float64[]
    ψ2_norms = Float64[]
    start_time = 0.0
  end


  for t in start_time:δt:ttotal
    tick()
    println(maxlinkdim(ψ))
    println(maxlinkdim(ψ2))

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
    println(corr[c])
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

    ψ = apply(real_gates, ψ; cutoff, maxdim)
    GC.gc()
    # normalize!(ψ)
    ψ2 = apply(real_gates, ψ2; cutoff, maxdim)
    GC.gc()
    # normalize!(ψ2)

    tock()
  end

  # plt.loglog(times, abs.(corrs))
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