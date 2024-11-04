using ITensors

function hubbard_model(L, t, U)
    os = OpSum()

    for i in 1:L-1
        os += -t, "Cdagup", i, "Cup", i + 1
        os += -t, "Cdagup", i + 1, "Cup", i
        os += -t, "Cdagdn", i, "Cdn", i + 1
        os += -t, "Cdagdn", i + 1, "Cdn", i
        
        os += U, "Nup", i, "Ndn", i
    end
    os += U, "Nup", L, "Ndn", L

    return os
end

L = 98
t = 1.0
U = -4.0

nsweeps = 10
maxdim = 300
cutoff = 1e-10

sites = siteinds("Electron", L; conserve_qns=true)
H = MPO(hubbard_model(L, t, U), sites)

N = div(L,2)
state = ["Emp" for n=1:L]
for i in 1:N
    if isodd(i)
        state[2*i-1] = "Up"
    else
        state[2*i-1] = "Dn"
    end
end
state[N] = "Emp"
ψ0 = MPS(sites, state)
E_N, ψ = dmrg(H, ψ0; nsweeps, maxdim, cutoff)

state[N] = "Up"
ψ0 = MPS(sites, state)
E_N1, ψ = dmrg(H, ψ0; nsweeps, maxdim, cutoff)

state[N+1] = "Dn"
ψ0 = MPS(sites, state)
E_N2, ψ = dmrg(H, ψ0; nsweeps, maxdim, cutoff)

ΔE = (2 * E_N1) - E_N - E_N2
print(ΔE)