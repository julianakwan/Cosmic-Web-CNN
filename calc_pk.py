import numpy as np
import matplotlib.pyplot as plt
import readbahamas
import Pk_library as PKL
import MAS_library as MASL
import sys
import camb
from camb import model, initialpower
from camb.dark_energy import DarkEnergyPPF, DarkEnergyFluid

def calculate_density(model, BoxSize):
    nfiles = int(16)     #There are 16 files per snapshot
    Ngrid = 1600
    MAS = 'CIC'
    snap_no = 32
    
    pos = []

    for i in range(0,nfiles):
        dir = '/beegfs2/hpcdata0_backup/simulations/BAHAMAS/DARK_ENERGY_CONSTANT_W/L400N1024/w_{:s}/data/snapshot_{:0>3d}'.format(model, int(snap_no))
#        dir = '/beegfs2/hpcdata0_backup/simulations/BAHAMAS/DMONLY_nu0_L400N1024_WMAP9/Data/Snapshots/snapshot_{:0>3d}'.format(int(snap_no))        
        filename = dir+'/snap_{:0>3d}.{:d}.hdf5'.format(int(snap_no),i)

#        #Read the particle co-ords
        if (i==0):
            pos, _ = readbahamas.get_coords(filename,1)
        else:
            pos_tmp, _ = readbahamas.get_coords(filename,1)
            pos = np.concatenate([pos,pos_tmp],axis=0)




    dens = np.zeros([Ngrid, Ngrid, Ngrid],dtype=np.float32)

    MASL.MA(pos, dens, BoxSize, MAS, verbose=True)

    dens /= np.mean(dens, dtype=np.float64)
    dens -= 1

    return dens

def calculate_pk(delta, BoxSize):
    MAS = 'CIC'
    axis = 0
    threads = 1
    
    Pk = PKL.Pk(delta,BoxSize,axis,MAS,threads,verbose=True)

    np.savetxt("/hpcdata2/arijkwan/masters/pk_{:s}.txt".format(model), np.column_stack([Pk.k3D, Pk.Pk[:,0]]))

    return Pk


if __name__=='__main__':
#    model = sys.argv[1]
    
    #read pre-calculated power spectra from density cubes

    k, pk_m0p5, pk_m1p0, pk_m1p5, pk_m2p0 = np.loadtxt("/hpcdata2/arijkwan/masters/density_pk.txt").T

    #read pre-calculated power spectra from snapshots
    k_m0p5_test, pk_m0p5_test = np.loadtxt("/hpcdata2/arijkwan/masters/pk_m0p5.txt").T
    k_m1p0_test, pk_m1p0_test = np.loadtxt("/hpcdata2/arijkwan/masters/pk_m1p0.txt").T
    k_m1p5_test, pk_m1p5_test = np.loadtxt("/hpcdata2/arijkwan/masters/pk_m1p5.txt").T
    k_m2p0_test, pk_m2p0_test = np.loadtxt("/hpcdata2/arijkwan/masters/pk_m2p0.txt").T


    #get linear theory predictions from camb
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.11, ombh2=0.0221, omch2=0.1209, mnu=0.0, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2.14e-9,ns=0.9624, r=0)  #sigma8 = 0.8349
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    pars.set_matter_power(redshifts=[0.], kmax=2.0)
    pars.NonLinear=model.NonLinear_none
    results = camb.get_results(pars)
    k_m1p0_camb, z, pk_m1p0_camb = results.get_matter_power_spectrum(minkh=1e-3, maxkh=1, npoints = 200)
    print(results.get_sigma8())

    #w = -0.5
    pars.DarkEnergy = DarkEnergyPPF(w=-0.5, wa=0.)
    results = camb.get_results(pars)
    k_m0p5_camb, z, pk_m0p5_camb = results.get_matter_power_spectrum(minkh=1e-3, maxkh=1, npoints = 200)
    print(results.get_sigma8())

    
    #w = -1.5
    pars.DarkEnergy = DarkEnergyPPF(w=-1.5, wa=0.)
    results = camb.get_results(pars)
    k_m1p5_camb, z, pk_m1p5_camb = results.get_matter_power_spectrum(minkh=1e-3, maxkh=1, npoints = 200)
    print(results.get_sigma8())
    
    #w = -2.0
    pars.DarkEnergy = DarkEnergyPPF(w=-2.0, wa=0.)
    results = camb.get_results(pars)
    k_m2p0_camb, z, pk_m2p0_camb = results.get_matter_power_spectrum(minkh=1e-3, maxkh=1, npoints = 200)
    print(results.get_sigma8())
    
    plt.loglog(k, pk_m0p5, color='blue', label = 'w=-0.5')
    plt.loglog(k, pk_m1p0, color='red', label = 'w=-1.0')
    plt.loglog(k, pk_m1p5, color='green', label = 'w=-1.5')
    plt.loglog(k, pk_m2p0, color ='cyan', label = 'w=-2.0')

    plt.loglog(k_m0p5_test, pk_m0p5_test, color='blue', linestyle='dashed')
    plt.loglog(k_m1p0_test, pk_m1p0_test, color='red', linestyle='dashed')
    plt.loglog(k_m1p5_test, pk_m1p5_test, color='green', linestyle='dashed')
    plt.loglog(k_m2p0_test, pk_m2p0_test, color='cyan', linestyle='dashed')

    plt.loglog(k_m0p5_camb, pk_m0p5_camb.reshape(-1,), color='blue', linestyle='dotted')
    plt.loglog(k_m1p0_camb, pk_m1p0_camb.reshape(-1,), color='red', linestyle='dotted')
    plt.loglog(k_m1p5_camb, pk_m1p5_camb.reshape(-1,), color='green', linestyle='dotted')
    plt.loglog(k_m2p0_camb, pk_m2p0_camb.reshape(-1,), color='cyan', linestyle='dotted')


    plt.legend(loc='best')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel(r'P(k) [Mpc/h]$^3$')
    plt.savefig("/hpcdata2/arijkwan/masters/pk_density_map_check.png")
    
    
