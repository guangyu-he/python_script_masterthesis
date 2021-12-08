#########################################
#ANCHOR Python script for Master Thesis
#Thesis Title: Determining Regularization and Simulating the Electron Beam in Non-translational Ptychography
#Author: Guangyu He@HU Berlin, Marcel Scholz@HU Berlin
#ANCHOR sub script for providing a standard test beam
#########################################

import numpy as np
from matplotlib import pyplot as plt

def multiplyLensFunction(psi, Dimension, wavelength, PixelSize):
    dim1 = Dimension
    dim2 = Dimension

    for i1 in range(-int(dim1/2), int(dim1/2)):
        for i2 in range(-int(dim2/2), int(dim2/2)):

            nu1 = (i1 / dim1) * (wavelength / PixelSize)
            nu2 = (i2 / dim2) * (wavelength / PixelSize)
            phi = np.arctan2(nu2, nu1)
            nu = np.sqrt(nu1 * nu1 + nu2 * nu2)

            if (nu < 0.0214):
                
                W = nu * nu * ( 0.5 * ( 0 * np.cos( 2.0 * ( phi - 0)) + 0 + 0 - (1 + 1) * 5.0e-10 * 0.5) + nu * ( 0.33333333 * ( 0 * np.cos( 3.0 * ( phi - 0)) + 0 * np.cos( phi - 0)) + nu * ( 0.25 * ( 0 * np.cos( 4.0 * ( phi - 0)) + 0 * np.cos( 2.0 * ( phi - 0) ) + 0) + nu * ( 0.2 * ( 0 * np.cos( 5.0 * ( phi - 0) ) + 0 * np.cos( phi - 0) + 0 * np.cos( 3.0 * ( phi - 0 ) ) ) + nu * ( 0.1665 * ( 0 * np.cos( 6.0 * ( phi - 0 ) ) + 0 * np.cos( 4.0 * ( phi - 0 ) ) + 0 * np.cos( 2.0 * ( phi - 0 ) ) + 0) ) ) ) ) )
                damp = 0 * nu * nu / wavelength
                damp = np.exp( - 2.0 * damp * damp )
                phi = damp * np.cos( 2.0 * np.pi * ( W / wavelength ) )
                damp = damp * np.sin( - 2.0 * np.pi * ( W / wavelength ) )
                re = np.real(psi[i1, i2])
                im = np.imag(psi[i1, i2])
                psi[i1, i2] = (phi * re - damp * im) + (phi * im + damp * re) * 1j 

            else:
                psi[i1, i2] = 0.0 + 0.0 * 1j

    #print(psi[144,144])

    return psi

def incomingWave(psi):
    psi += 1.0 + 0.0 * 1j
    psi = multiplyLensFunction(psi, 144, 4.866e-12, 0.135e-10)

    psi = np.fft.fft2(psi) 
    psi = np.fft.fftshift(psi)
    psi /= np.sqrt( np.sum( np.real(psi) * np.real(psi)) + np.sum( np.imag(psi) * np.imag(psi)) ) 
    #plt.clf(),plt.imshow(np.abs(psi),cmap='jet',interpolation='None',aspect='1'),plt.show()
    
    return psi

def standardbeam():
    psi = np.zeros( (4, 144, 144), dtype=complex)
    psi[0, :, :] = incomingWave(psi[0, :, :])
    return psi[0, :, :]

if __name__ == "__main__":

    standardbeam()