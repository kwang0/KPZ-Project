from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.purification_mps import PurificationMPS
from tenpy.algorithms.purification import PurificationTEBD, PurificationApplyMPO
from tenpy.networks.mps import MPSEnvironment
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

# Purification to finite temperature of beta_max (Schollwock 7.2)
def imag_tebd(L=128, beta_max=3., dt=0.05, chi_max=256, order=2, bc="finite"):
    model_params = dict(L=L, Jxx=1., Jz=1., hz=0, sort_charge=True)
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

# Real time evolution using MPO
def real_mpo(psi, model, N_steps = 1000, dt = 0.1, chi_max = 256, beta = 3):
    options = {
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1.e-8
        }
    }

    Us = [model.H_MPO.make_U(-d * 1.j * dt, 'II') for d in [0.5 + 0.5j, 0.5 - 0.5j]]
    mpo_evolve = PurificationApplyMPO(psi, Us[0], options)

    L = psi.L
    psi2 = psi.copy()
    psi2.apply_local_op(L//2,'Sz') # Time-evolve Sz|psi> and |psi> separately
    mpo2_evolve = PurificationApplyMPO(psi2, Us[0], options)

    times = []
    corrs = []
    for i in range(N_steps):
        for U in Us:
            mpo_evolve.init_env(U)
            mpo_evolve.run()

            mpo2_evolve.init_env(U)
            mpo2_evolve.run()

        env = MPSEnvironment(psi, psi2)

        times.append(i * dt)
        print('t =',i * dt)
        corrs.append(np.abs(env.expectation_value('Sz', sites=[L//2])[0]))

        # beta = 3
        # Writing to data file. Beta and dt are inserted in the filename to one sig fig to avoid decimal
        # points in scientific notation.
        with h5py.File("data/mpo_L{}_chi{}_beta{:.0e}_dt{:.0e}.h5".format(L, chi_max, beta, dt),'w') as F:
            F['times'] = times
            F['corrs'] = corrs

    return times, corrs

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Pass in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--length", type=int, help = "Length of chain")
    parser.add_argument("-b", "--beta", type=float, help="Inverse temperature: beta")
    parser.add_argument("-c", "--chi", type=int, help="Maximum bond dimension: chi_max")
    parser.add_argument("-t", "--dt", type=float, help="Real-time evolution timestep: dt")
    args = parser.parse_args()
    L = args.length
    beta = args.beta
    chi_max = args.chi
    dt = args.dt

    psi, model = imag_tebd(beta_max=beta, L=L)
    times, corrs = real_mpo(psi, model, chi_max=chi_max, N_steps=1000, dt=dt, beta=beta)


    # plt.loglog(times, corrs)
    # plt.xlabel(r'$t$')
    # plt.ylabel(r'$|C(T,x=0,t)|$')
    # plt.show()







