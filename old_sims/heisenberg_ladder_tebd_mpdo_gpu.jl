using MKL
using LinearAlgebra
using ITensors
using NDTensors
using CUDA
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
  
  return NDTensors.cu(exp(-im * t * (J1 * (h1 + h2) + J2 * rung)))
end

function current(j)
  os = OpSum()

  os += 0.5 * im, "S1+", j, "S1-", j + 1
  os += -0.5 * im, "S1-", j, "S1+", j + 1

  os += 0.5 * im, "S2+", j, "S2-", j + 1
  os += -0.5 * im, "S2-", j, "S2+", j + 1

  return os
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

function fourth_order_trotter_gates(L, sites, δt, J1, J2)
  a1 = 0.095848502741203681182
  a2 = -0.078111158921637922695
  a3 = 0.5 - (a1 + a2)
  b1 = 0.42652466131587616168
  b2 = -0.12039526945509726545
  b3 = 1 - 2 * (b1 + b2)

  A1 = ops([("expiSS", (n, n+1), (t=a1*δt, J1=J1, J2=J2,)) for n in 1:2:(L - 1)], sites)
  B1 = ops([("expiSS", (n, n+1), (t=b1*δt, J1=J1, J2=J2,)) for n in 2:2:(L - 2)], sites)
  A2 = ops([("expiSS", (n, n+1), (t=a2*δt, J1=J1, J2=J2,)) for n in 1:2:(L - 1)], sites)
  B2 = ops([("expiSS", (n, n+1), (t=b2*δt, J1=J1, J2=J2,)) for n in 2:2:(L - 2)], sites)
  A3 = ops([("expiSS", (n, n+1), (t=a3*δt, J1=J1, J2=J2,)) for n in 1:2:(L - 1)], sites)
  B3 = ops([("expiSS", (n, n+1), (t=b3*δt, J1=J1, J2=J2,)) for n in 2:2:(L - 2)], sites)

  return vcat(A1,B1,A2,B2,A3,B3,A3,B2,A2,B1,A1)
end

function main(; L=128, cutoff=1E-10, δt=0.1, ttotal=100, maxdim=32, J1=1.0, J2=0.0, μ=0.001)

  c = div(L,2) + 1 # center site

  filename = "/pscratch/sd/k/kwang98/KPZ/tebd_mpdo_gpu_L$(L)_chi$(maxdim)_dt$(δt)_Jprime$(J2)_mu$(μ).h5"
  # filename = "tebd_mpdo_gpu_L$(L)_chi$(maxdim)_dt$(δt)_Jprime$(J2)_mu$(μ).h5"
  if (isfile(filename))
    F = h5open(filename,"r")
    times = read(F, "times")
    Z1s = read(F, "Z1s")
    Z2s = read(F, "Z2s")
    Js = read(F, "Js")
    ρ = NDTensors.cu(read(F, "rho", MPO))
    sites = read(F,"sites",Vector{Index{Int64}})
    start_time = last(times)
    close(F)

    # Make real-time evolution gates
    real_gates = fourth_order_trotter_gates(L, sites, δt, J1, J2)
    GC.gc()
  else
    sites = siteinds("S=3/2", L; conserve_qns=false)

    # Initial state is weakly polarized domain-wall mixed state
    ρ = domain_wall(L, sites, μ)
    real_gates = fourth_order_trotter_gates(L, sites, δt, J1, J2)
    GC.gc()
    
    orthogonalize!(ρ, c)
    ρ = NDTensors.cu(ρ)
    # normalize!(ψ2)
  
    times = Float64[]
    Z1s = []
    Z2s = []
    Js = []
    start_time = 0.0
  end


  for t in start_time:δt:ttotal
    println(maxlinkdim(ρ))

    Z1 = ComplexF64[]
    Z2 = ComplexF64[]
    J = ComplexF64[]
    println("Copying to cpu to calculate expectation values")
    @time ρ = NDTensors.cpu(ρ)
    for i in 1:L
      orthogonalize!(ρ, i)
      S1z = 2 * op("S1z",sites[i])
      S2z = 2 * op("S2z",sites[i])
      push!(Z1, tr(apply(S1z, ρ; cutoff, maxdim)))
      push!(Z2, tr(apply(S2z, ρ; cutoff, maxdim)))
      if i < L
        j = MPO(current(i), sites)
        push!(J, tr(apply(j, ρ; cutoff, maxdim)))
      end
    end
    orthogonalize!(ρ, c)
    println("Time = $t")
    flush(stdout)
    push!(times, t)
    t == 0.0 ? Z1s = Z1 : Z1s = hcat(Z1s, Z1)
    t == 0.0 ? Z2s = Z2 : Z2s = hcat(Z2s, Z2)
    t == 0.0 ? Js = J : Js = hcat(Js, J)

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

    println("Copying back to gpu to apply gates")
    @time ρ = NDTensors.cu(ρ)
    println("Applying gates")
    @time ρ = apply(real_gates, ρ; cutoff, maxdim, apply_dag=true)
    GC.gc()
  end
end

# Set to identity to run on CPU
# gpu = cu

ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(1)
# ITensors.enable_threaded_blocksparse(true)

L = parse(Int64, ARGS[1])
maxdim = parse(Int64, ARGS[2])
δt = parse(Float64, ARGS[3])
J2 = parse(Float64, ARGS[4])
μ = parse(Float64, ARGS[5])

main(L=L, maxdim=maxdim, δt=δt, J2=J2, μ=μ)