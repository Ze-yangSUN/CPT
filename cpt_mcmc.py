import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import emcee

lmax = L = 1024; lmin = 2; nside = 512; Nf = 6
ell = np.arange(lmax+1)
fre = np.array([50, 78, 119, 195, 235, 337])

def log_likelihood(theta, cl_EEo, cl_BBo, cl_EBo, EEcmb_bl2, BBcmb_bl2):
    alpha, beta = theta
    model = 0.5*np.tan(4*alpha)*(EEo-BBo) + 0.5*np.sin(4*beta)/np.cos(4*alpha)*(EEcmb_bl2-BBcmb_bl2)
    var = (1./(2.*ell+1)*EEo*BBo+ 0.5*(np.tan(4*alpha))**2/(2.*ell+1)*(EEo**2+BBo**2) - np.tan(4*alpha)*2./(2.*ell+1)*EBo*(EEo-BBo)) + 1./(2.*ell+1)*(1-(np.tan(4*alpha))**2)*EBo**2
    return -np.sum( (EBo[lmin:L]-model[lmin:L])**2/var[lmin:L] + 2.*np.pi*np.log(var[lmin:L]) )

def log_prior(theta):
    alpha, beta = theta
    if (-10.0*np.pi/180. < alpha < 10.0*np.pi/180.) and (-10.0*np.pi/180. < beta < 10.0*np.pi/180.):
        return 0.0
    return -np.inf

def log_probability(theta, cl_EEo, cl_BBo, cl_EBo, EEcmb_bl2, BBcmb_bl2):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, cl_EEo, cl_BBo, cl_EBo, EEcmb_bl2, BBcmb_bl2)    

loc = 'rec_angle/version1/1st/fig1/data/'
cl_EEo = np.load('%s/cl_EEo_fig4.npy'%loc)
cl_BBo = np.load('%s/cl_BBo_fig4.npy'%loc)
cl_EBo = np.load('%s/cl_EBo_fig4.npy'%loc)
cl_EEcmb_bl2 = np.load('%s/cl_EEth_bl2.npy'%loc, )
cl_BBcmb_bl2 = np.load('%s/cl_BBth_bl2.npy'%loc, )

i = 0
j = 0
EEo, BBo, EBo, EEcmb_bl2, BBcmb_bl2= cl_EEo[i][j], cl_BBo[i][j], cl_EBo[i][j], cl_EEcmb_bl2[i], cl_BBcmb_bl2[i]

deg = np.array([-3, -2, -1, 0, 1, 2, 3])*np.pi/180       
pos = np.array([0.0, 0.5*np.pi/180 ]) + np.random.uniform(-10,10,(20,2)) * np.pi/180.

# initialize the walkers
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(cl_EEo, cl_BBo, cl_EBo, EEcmb_bl2, BBcmb_bl2))
pos, log_prob, state = sampler.run_mcmc(pos, 30, progress = True)

# plot the positions of each walker as a function of the number of steps in the chain:
fig, axes = plt.subplots(2, figsize=(10, 5), sharex=True)
samples = sampler.get_chain()
labels = ["alpha", "beta"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:,:,i]*180/np.pi, 'k',)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel("step number");

# reset sampler
sampler.reset()
sampler.run_mcmc(pos, 2000, progress = True)
flat_samples = sampler.get_chain(discard=500, thin=1, flat=True)
