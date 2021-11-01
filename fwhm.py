#########################################
#python script for Master Thesis
#Determining Regularization and Simulating the Electron Beam in Non-translational Ptychography
#Guangyu He 
#########################################

from __future__ import division 
import numpy as np
import matplotlib.pyplot as plt
import math

def interpolated_intercepts(x, y1, y2):
    """Find the intercepts of two curves, given by the same x data"""

    def intercept(point1, point2, point3, point4):
        """find the intersection between two lines
        the first line is defined by the line between point1 and point2
        the first line is defined by the line between point3 and point4
        each point is an (x,y) tuple.

        So, for example, you can find the intersection between
        intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

        Returns: the intercept, in (x,y) format
        """    

        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x,y

        L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
        L2 = line([point3[0],point3[1]], [point4[0],point4[1]])

        R = intersection(L1, L2)

        return R

    idxs = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)

    xcs = []
    ycs = []

    for idx in idxs:
        xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
        xcs.append(xc)
        ycs.append(yc)
    return np.array(xcs), np.array(ycs)

def fwhm(inten):

    """
    plot the intensity distribution on profile line, 
    and return the width of fwhm
    """

    len_profile = inten.shape[0] #length of profile
  
    x = np.arange(0,len_profile,1)
    profile = inten[:,math.ceil(inten.shape[1]/2)]
    y2 = 0 * x + profile.max()/2 #high max line

    #plt.clf()
    #plt.plot(x, profile, marker='', mec='none', ms=4, lw=1, label='inten. dist. on profile')
    #plt.plot(x, y2, marker='o', mec='none', ms=4, lw=1, label='half max line')

    idx = np.argwhere(np.diff(np.sign(profile - y2)) != 0)

    #plt.plot(x[idx], profile[idx], 'ms', ms=7)#, label='Nearest data-point method')

    # new method!
    xcs, ycs = interpolated_intercepts(x,profile,y2)

    xcmax = 0
    xcmin = len_profile

    for xc, yc in zip(xcs, ycs):
        #plt.plot(xc, yc, 'co', ms=5)#, label='Nearest data-point, with linear interpolation')
        if xc>xcmax:
            xcmax = xc
        if xc<xcmin:
            xcmin = xc
    
    peak_width = xcmax - xcmin
    peak_width = math.ceil(peak_width)


    #plt.legend(frameon=False, fontsize=8, numpoints=1, loc='lower left')
    #plt.title('Intensity distribution on profile line')
    #plt.savefig('fig/curve crossing.png', dpi=200)

    return peak_width
