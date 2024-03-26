using MKL
using LinearAlgebra
using ITensors
using Printf
using PyPlot
using HDF5
using TickTock

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

function ITensors.op(::OpName"expiSS", ::SiteType"S=3/2", s1::Index, s2::Index; t, J1, J2, U1)
  h1 =
    1 / 2 * op("S1+", s1) * op("S1-", s2) +
    1 / 2 * op("S1-", s1) * op("S1+", s2) +
    op("S1z", s1) * op("S1z", s2)
  h2 =
    1 / 2 * op("S2+", s1) * op("S2-", s2) +
    1 / 2 * op("S2-", s1) * op("S2+", s2) +
    op("S2z", s1) * op("S2z", s2)
  rung = op("rung", s1) * op("Id", s2)
  biquad = setprime(h1 * prime(h2), 1; plev=2)
  
  return exp(-im * t * (J1 * (h1 + h2) + J2 * rung + U1 * biquad))
end

function ITensors.op(::OpName"current", ::SiteType"S=3/2", s1::Index, s2::Index)
  cur = 
    0.5 * im * op("S1+", s1) * op("S1-", s2) -
    0.5 * im * op("S1-", s1) * op("S1+", s2) +
    0.5 * im * op("S2+", s1) * op("S2-", s2) -
    0.5 * im * op("S2-", s1) * op("S2+", s2)

  return cur
end

function domain_wall(L, sites, μ)
  ρ = MPO(L)
  for i in 1:L
    if i < div(L,2) + 1
      ρ[i] = op("Id", sites[i]) + 2 * μ * op("S1z", sites[i]) + 2 * μ * op("S2z", sites[i])
    else
      ρ[i] = op("Id", sites[i]) - 2 * μ * op("S1z", sites[i]) - 2 * μ * op("S2z", sites[i])
    end
    ρ[i] ./= tr(ρ[i])
  end
  return ρ
end

function fourth_order_trotter_gates(L, sites, δt, J1, J2, U1)
  a1 = 0.095848502741203681182
  a2 = -0.078111158921637922695
  a3 = 0.5 - (a1 + a2)
  b1 = 0.42652466131587616168
  b2 = -0.12039526945509726545
  b3 = 1 - 2 * (b1 + b2)

  A1 = ops([("expiSS", (n, n+1), (t=a1*δt, J1=J1, J2=J2, U1=U1,)) for n in 1:2:(L - 1)], sites)
  B1 = ops([("expiSS", (n, n+1), (t=b1*δt, J1=J1, J2=J2, U1=U1,)) for n in 2:2:(L - 2)], sites)
  A2 = ops([("expiSS", (n, n+1), (t=a2*δt, J1=J1, J2=J2, U1=U1,)) for n in 1:2:(L - 1)], sites)
  B2 = ops([("expiSS", (n, n+1), (t=b2*δt, J1=J1, J2=J2, U1=U1,)) for n in 2:2:(L - 2)], sites)
  A3 = ops([("expiSS", (n, n+1), (t=a3*δt, J1=J1, J2=J2, U1=U1,)) for n in 1:2:(L - 1)], sites)
  B3 = ops([("expiSS", (n, n+1), (t=b3*δt, J1=J1, J2=J2, U1=U1,)) for n in 2:2:(L - 2)], sites)

  return vcat(A1,B1,A2,B2,A3,B3,A3,B2,A2,B1,A1)
end

function main(; L=128, cutoff=1E-10, δt=0.1, ttotal=100, maxdim=32, J1=1.0, J2=0.0, U1=0.0, μ=0.001)
  tick()

  c = div(L,2) + 1 # center site

  filename = "/pscratch/sd/k/kwang98/KPZ/tebd_mpdo_L$(L)_chi$(maxdim)_dt$(δt)_Jprime$(J2)_U$(U1)_mu$(μ).h5"
  # filename = "/global/scratch/users/kwang98/tebd_mpdo_L$(L)_chi$(maxdim)_dt$(δt)_Jprime$(J2)_U$(U1)_mu$(μ).h5"
  # filename = "tebd_mpdo_L$(L)_chi$(maxdim)_dt$(δt)_Jprime$(J2)_U$(U1)_mu$(μ).h5"
  if (isfile(filename))
    F = h5open(filename,"r")
    times = read(F, "times")
    Z1s = read(F, "Z1s")
    Z2s = read(F, "Z2s")
    Js = read(F, "Js")
    ρ = read(F, "rho", MPO)
    sites = read(F,"sites",Vector{Index{Vector{Pair{QN, Int64}}}})
    start_time = last(times) + δt
    close(F)

    # Make real-time evolution gates
    real_gates = fourth_order_trotter_gates(L, sites, δt, J1, J2, U1)
    GC.gc()
  else
    sites = siteinds("S=3/2", L; conserve_qns=true)

    # Initial state is weakly polarized domain-wall mixed state
    ρ = domain_wall(L, sites, μ)
    real_gates = fourth_order_trotter_gates(L, sites, δt, J1, J2, U1)
    GC.gc()
    
    orthogonalize!(ρ, c)
    # normalize!(ψ2)
  
    times = Float64[]
    Z1s = []
    Z2s = []
    Js = []
    start_time = δt
  end


  for t in start_time:δt:ttotal
    # Stop simulations before HPC limit to ensure no corruption of data writing
    if peektimer() > (23.5 * 60 * 60)
      break
    end

    println(maxlinkdim(ρ))

    @time ρ = apply(real_gates, ρ; cutoff, maxdim, apply_dag=true)
    GC.gc()

    Z1 = ComplexF64[]
    Z2 = ComplexF64[]
    J = ComplexF64[]
    @time for i in 1:L
      # orthogonalize!(ρ, i)
      S1z = 2 * op("S1z",sites[i])
      S2z = 2 * op("S2z",sites[i])
      push!(Z1, tr(apply(S1z, ρ; cutoff, maxdim)))
      push!(Z2, tr(apply(S2z, ρ; cutoff, maxdim)))
      if i < L
        current = op("current", sites[i], sites[i+1])
        push!(J, tr(apply(current, ρ; cutoff, maxdim)))
      end
    end
    # orthogonalize!(ρ, c)
    println("Time = $t")
    flush(stdout)
    push!(times, t)
    t == δt ? Z1s = Z1 : Z1s = hcat(Z1s, Z1)
    t == δt ? Z2s = Z2 : Z2s = hcat(Z2s, Z2)
    t == δt ? Js = J : Js = hcat(Js, J)

    # Writing to data file
    F = h5open(filename,"w")
    F["times"] = times
    F["Z1s"] = Z1s
    F["Z2s"] = Z2s
    F["Js"] = Js
    F["corrs"] = (Z1s[c-1,:] .- Z1s[c,:]) ./ (2 * μ)
    F["rho"] = ρ
    F["sites"] = sites
    close(F)

    t≈ttotal && break
  end
end


ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(256)
# ITensors.enable_threaded_blocksparse(true)

J1 = 1.0

L = parse(Int64, ARGS[1])
maxdim = parse(Int64, ARGS[2])
δt = parse(Float64, ARGS[3])
J2 = parse(Float64, ARGS[4])
U1 = parse(Float64, ARGS[5])
μ = parse(Float64, ARGS[6])

main(L=L, maxdim=maxdim, δt=δt, J1=J1, J2=J2, U1=U1, μ=μ)