from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.purification_mps import PurificationMPS
from tenpy.algorithms.purification import PurificationTEBD, PurificationApplyMPO
from tenpy.algorithms import tdvp
import copy
import numpy as np
import matplotlib.pyplot as plt

def imag_tebd(L=128, beta_max=3., dt=0.05, order=2, bc="finite"):
    model_params = dict(L=L, Jxx=1., Jz=1., hz=0)
    M = XXZChain(model_params)
    psi = PurificationMPS.from_infiniteT(M.lat.mps_sites(), bc=bc)
    options = {
        'trunc_params': {
            'chi_max': 256,
            'svd_min': 1.e-8
        },
        'order': order,
        'dt': dt,
        'N_steps': 1
    }
    beta = 0.
    eng = PurificationTEBD(psi, M, options)
    while beta < beta_max:
        beta += 2. * dt  # factor of 2:  |psi> ~= exp^{- dt H}, but rho = |psi><psi|
        eng.run_imaginary(dt)  # cool down by dt
    
    psi2 = psi.copy()
    psi2.apply_local_op(0,'Sz') # Time-evolve Sz|psi> and |psi> separately
    eng2 = PurificationTEBD(psi2, M, options)

    times = []
    corrs = []
    for i in range(200):
        eng.run_evolution(1,0.5)
        eng2.run_evolution(1,0.5)

        psi3 = psi2.copy()
        psi3.apply_local_op(0,'Sz')

        times.append(eng.evolved_time)
        print(eng.evolved_time)
        corrs.append(np.abs(psi.overlap(psi3)))

    return times, corrs

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    times, corrs = imag_tebd()

    plt.loglog(times, corrs)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$|C(T,x=0,t)|$')
    plt.show()







