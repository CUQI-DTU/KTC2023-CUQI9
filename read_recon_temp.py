# read files in Output folder

#%%
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
Nel = 32  # number of electrodes
from skimage.segmentation import chan_vese
from EITLib.segmentation import scoring_function

#%%
def cv(deltareco_pixgrid, log_par=1.5, linear_par=1, exp_par=0):
        mu = np.mean(deltareco_pixgrid)
        
        
        cv = chan_vese(linear_par*deltareco_pixgrid + log_par*np.abs(np.log(deltareco_pixgrid) + exp_par*np.exp(deltareco_pixgrid)), mu=0.1, lambda1=1, lambda2=1, tol=1e-6,
                    max_num_iter=1000, dt=2.5, init_level_set="checkerboard",
                    extended_output=True)

        labeled_array, num_features = sp.ndimage.label(cv[0])
        # Initialize a list to store masks for each region
        region_masks = []

        # Loop through each labeled region
        deltareco_pixgrid_segmented = np.zeros((256,256))

        for label in range(1, num_features + 1):
            # Create a mask for the current region
            region_mask = labeled_array == label
            region_masks.append(region_mask)
            if np.mean(deltareco_pixgrid[region_mask]) < mu:
                deltareco_pixgrid_segmented[region_mask] = 1
            else:
                deltareco_pixgrid_segmented[region_mask] = 2
        return deltareco_pixgrid_segmented

#%%
for i in range(4,5):
    recon_file = sp.io.loadmat('Output1/' + str(i) +'.mat')


    plt.figure()
    im = plt.imshow(recon_file['reconstruction'])
    plt.title('reconstruction '+str(i))
    plt.colorbar(im)

    # plot the true conductivity
    plt.figure()
    phantom_file = sp.io.loadmat('GroundTruths/true'+str(i)+'.mat')
    im = plt.imshow(phantom_file['truth'])
    plt.title('true conductivity '+str(i))
    plt.colorbar(im)

    # load original reconstruction
    plt.figure()
    orig_recon = np.load('Output1/' + str(i) +'.npz')['deltareco_pixgrid']
    im = plt.imshow(np.log(orig_recon))
    plt.title('log orig recon conductivity '+str(i))
    plt.colorbar(im)

    # load KTC challange recon

    # segment with chan-vese
    plt.figure()
    seg = cv(orig_recon, log_par=1.5, linear_par=0, exp_par=0)
    im = plt.imshow(seg)
    plt.colorbar(im)
    plt.title('chan-vese segmentation '+str(i))
    
    print(i)
    print(scoring_function(phantom_file['truth'], seg))

    

# %%
