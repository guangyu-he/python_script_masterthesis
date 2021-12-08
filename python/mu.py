#########################################
#ANCHOR Python script for Master Thesis
#Thesis Title: Determining Regularization and Simulating the Electron Beam in Non-translational Ptychography
#Author: Guangyu He@HU Berlin
#ANCHOR sub script for creating diffraction patterns with poisson noise and calculate the fraction R
#########################################

import numpy as np
import struct
from matplotlib import pyplot as plt
from numpy.lib.type_check import imag


from bin import bin_pack,bin_unpack

#Dimension of CBED
dimension = 96

#path of diffraction patterns
path = "bin/DPS_sum.bin"

#gain gamma and Poisson Noise
gamma = 10000

#display which probe
probe_nr = 1

image_mode = True
create_poisson_mode = True
r_mode = True

if create_poisson_mode:
    r_mode = False

def drawcbed(path,Dimension,cbed):
    f = open(path, 'rb')
    f.seek(Dimension *Dimension*4 * cbed)
    data = f.read(Dimension *Dimension*4)

    values=struct.unpack(Dimension*Dimension*'f', data)
    #Convert binary to numpy array
    values=np.asarray(values)
    #Reshape CBED into 2D
    values=np.reshape(values,(Dimension,Dimension))
    
    return values

def circle(value,radius):
    dim = len(value)
    # xx and yy are 200x200 tables containing the x and y coordinates as values
    # mgrid is a mesh creation helper
    xx, yy = np.mgrid[:dim, :dim]
    # circles contains the squared distance to the (100, 100) point
    # we are just using the circle equation learnt at school
    circle = (xx - dim/2) ** 2 + (yy - dim/2) ** 2
    # donuts contains 1's and 0's organized in a donut shape
    # you apply 2 thresholds on circle to define the shape
    donut = (circle < (radius**2 + 0))

    a = np.zeros(shape=(dim,dim))
    for x in range(dim):
        for y in range(dim):
            if donut[x,y]:
                a[x,y] = 1
            else:
                pass
    
    output = value * a

    return output

if create_poisson_mode:
    j_p = []
    for i in range(9):
        j = drawcbed(path,dimension,i)
        J = j * gamma
        J_p = np.random.poisson(J)
        j_p.append(J_p / gamma)

    j_p_numpy = np.array(j_p)
    data = np.ravel(j_p_numpy)
    data = struct.pack(9 * dimension * dimension *'f', *data)
    f = open('bin/DPS_sum_poisson.bin', 'wb')
    f.write(data)
    f.close()

if r_mode:
    one = np.ones([96,96])
    one = circle(one, 15)
    one_pixels = np.sum(one) * 9
    print(one_pixels)

    I_i = []
    J_i = []
    J0_i = []
    R = np.zeros(9)

    for i in range(9):
        J0_i_cbed = drawcbed("bin/DPS_sum.bin",dimension,i) * gamma
        J0_i_cbed = circle(J0_i_cbed, 15)
        J0_i.append(J0_i_cbed)
        
        I_i_cbed = drawcbed("mu_bin/Measurements_model75e-5.bin",dimension,i) * gamma
        I_i_cbed = circle(I_i_cbed, 15)
        I_i.append(I_i_cbed)

        J_i_cbed = drawcbed("bin/DPS_sum_poisson.bin",dimension,i) * gamma
        J_i_cbed = circle(J_i_cbed, 15)
        J_i.append(J_i_cbed)

        for m in range(dimension):
            for n in range(dimension):
                x = m #+ 33
                y = n #+ 33
                if I_i[i][x,y] == 0:
                    pass
                else:
                    #I_i[i][x,y] += 1e-1
                    R[i] += ( I_i[i][x,y] - J_i[i][x,y] )**2 / I_i[i][x,y]

    R_sum = np.sum(R)
    #print(R)
    print(R_sum)
    print("-----")
    print(one_pixels - R_sum)

if image_mode:
    plt.figure()

    if create_poisson_mode:
        plt.subplot(121),plt.imshow(np.abs(drawcbed("bin/DPS_sum.bin",96,0)),interpolation='None',cmap='jet',aspect='1')
        cb = plt.colorbar()
        plt.title('diffraction pattern')
        plt.subplot(122),plt.imshow(np.abs(drawcbed("bin/DPS_sum_poisson.bin",96,0)),interpolation='None',cmap='jet',aspect='1')
        cb = plt.colorbar()
        plt.title('diffraction pattern with noise')
    elif r_mode:
        plt.subplot(131),plt.imshow(np.abs(circle(J0_i[probe_nr], 15)),interpolation='None',cmap='jet',aspect='1')
        cb = plt.colorbar()
        plt.title('diffraction pattern')
        plt.subplot(132),plt.imshow(np.abs(circle(J_i[probe_nr], 15)),interpolation='None',cmap='jet',aspect='1')
        cb = plt.colorbar()
        plt.title('diffraction pattern with noise')
        plt.subplot(133),plt.imshow(np.abs(circle(I_i[probe_nr], 15)),interpolation='None',cmap='jet',aspect='1')
        cb = plt.colorbar()
        plt.title('reconstructed diffraction pattern without noise')
    else:
        pass

    plt.show()



