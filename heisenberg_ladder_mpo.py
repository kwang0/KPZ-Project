from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.purification_mps import PurificationMPS
from tenpy.algorithms.purification import PurificationTEBD, PurificationApplyMPO
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution
import copy
import numpy as np
import matplotlib.pyplot as plt

def imag_tebd(L=128, beta_max=3., dt=0.05, chi_max=256, order=2, bc="finite"):
    model_params = dict(L=L, Jxx=1., Jz=1., hz=0)
    model = XXZChain(model_params)
    psi = PurificationMPS.from_infiniteT(model.lat.mps_sites(), bc=bc)
    options = {
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1.e-8
        },
        'order': order,
        'dt': dt,
        'N_steps': 1
    }
    beta = 0.
    eng = PurificationTEBD(psi, model, options)
    while beta < beta_max:
        beta += 2. * dt  # factor of 2:  |psi> ~= exp^{- dt H}, but rho = |psi><psi|
        eng.run_imaginary(dt)  # cool down by dt
    
    return psi, model

def real_mpo(psi, model, N_steps = 1000, dt = 0.1, chi_max = 256):
    options = {
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1.e-8
        }
    }

    Us = [model.H_MPO.make_U(-d * 1.j * dt, 'II') for d in [0.5 + 0.5j, 0.5 - 0.5j]]
    mpo_evolve = PurificationApplyMPO(psi, Us[0], options)

    psi2 = psi.copy()
    psi2.apply_local_op(0,'Sz') # Time-evolve Sz|psi> and |psi> separately
    mpo2_evolve = PurificationApplyMPO(psi2, Us[0], options)

    times = []
    corrs = []
    for i in range(N_steps):
        for U in Us:
            mpo_evolve.init_env(U)
            mpo_evolve.run()

            mpo2_evolve.init_env(U)
            mpo2_evolve.run()

        psi3 = psi2.copy()
        psi3.apply_local_op(0,'Sz')

        times.append(i * dt)
        print(i * dt)
        corrs.append(np.abs(psi.overlap(psi3)))

    return times, corrs

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    psi, model = imag_tebd(L=32)
    times, corrs = real_mpo(psi, model, chi_max = 64)


    plt.loglog(times, corrs)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$|C(T,x=0,t)|$')
    plt.show()







