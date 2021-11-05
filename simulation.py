#########################################
#ANCHOR Python script for Master Thesis
#Thesis Title: Determining Regularization and Simulating the Electron Beam in Non-translational Ptychography
#Author: Guangyu He@HU Berlin
#ANCHOR sub script for beam simulation
#########################################

import numpy as np

def pre_simulation(inten,radius,width_pix,mat):

    """
    calculate the width of the beam
    """

    import math

    profile = inten[:,math.ceil(inten.shape[1]/2)]

    lam = 0.004866 #nm

    f = open("pre_width_info_"+mat+".txt", 'w+')

    half_diver_ang = 40 #normally in aperture, the radius is 40 mrad
    print("half divergence angle in Rec. space:" + str(half_diver_ang) + " mrad",file=f)

    pixel_ap = radius*2    #number of pixel of aperture's diameter
    print("number of pixel of diameter in Rec.space:" + str(pixel_ap),file=f)

    print("---------",file=f)

    m1 = np.size(profile,0) #m number of pixel of Rec. space
    #m=1510
    print("number of pixel in Rec. space:"+ str(m1),file=f)

    delta1_nm_1 = ( half_diver_ang * 2 * 0.001 / lam ) / pixel_ap 
    print("length of each pixel in Rec. space:"+ str(delta1_nm_1) + " nm^-1",file=f)

    d1 = 1 / (delta1_nm_1 * m1) #d1 each pixel length in real space
    print("length of each pixel in real space:"+ str(d1) + " nm",file=f)
    
    print("---------",file=f)

    width = width_pix * d1
    beam_area = math.pi * ( width / 2 )**2

    print("width in pixel:" + str(width_pix),file=f)
    print("width diameter:" + str(width) + " nm",file=f)
    print("beam_area:" + str(beam_area) + "nm²",file=f)

    print("---------",file=f)

    w = 6 * width
    delta2_nm_1 = 1 / w
    print("delta2:" + str(delta2_nm_1) + "nm^-1",file=f)

    n2 = 3 * 2 * half_diver_ang * 0.001 / lam / delta2_nm_1
    n2 = math.ceil(n2)
    if (n2 % 2) == 0:
        pass
    else:
        n2 = n2 + 1
    
    print("n2:" + str(n2) + "pixels",file=f)

    m2 = 1.5 * n2
    print("m2:" + str(m2) + "pixels",file=f)

    d2 = w / m2
    print("d2:" + str(d2) + "nm",file=f)

    f.close



    return m2,delta1_nm_1,delta2_nm_1

def simulation(inten,radius,width_pix,mat):

    """
    calculate the width of the beam
    """

    import math

    profile = inten[:,math.ceil(inten.shape[1]/2)]

    lam = 0.004866 #nm

    f = open("width_info_"+mat+".txt", 'w+')

    half_diver_ang = 40 #normally in aperture, the radius is 40 mrad
    print("half divergence angle in Rec. space:" + str(half_diver_ang) + " mrad",file=f)

    pixel_ap = radius*2    #number of pixel of aperture's diameter
    print("number of pixel of diameter in Rec.space:" + str(pixel_ap),file=f)

    print("---------",file=f)

    m1 = np.size(profile,0) #m number of pixel of Rec. space
    #m=1510
    print("number of pixel in Rec. space:"+ str(m1),file=f)

    delta1_nm_1 = ( half_diver_ang * 2 * 0.001 / lam ) / pixel_ap 
    print("length of each pixel in Rec. space:"+ str(delta1_nm_1) + " nm^-1",file=f)

    d1 = 1 / (delta1_nm_1 * m1) #d1 each pixel length in real space
    print("length of each pixel in real space:"+ str(d1) + " nm",file=f)
    
    print("---------",file=f)

    width = width_pix * d1
    beam_area = math.pi * ( width / 2 )**2

    print("width in pixel:" + str(width_pix),file=f)
    print("width diameter:" + str(width) + " nm",file=f)
    print("beam_area:" + str(beam_area) + "nm²",file=f)
    """
    print("---------",file=f)

    width_ideal = 0.61 * lam / (half_diver_ang * 0.001 / 2 )
    print("ideal width:" + str(width_ideal) + " nm",file=f)

    print("---------",file=f)


    w = 6 * width
    delta2_nm_1 = 1 / w
    print("delta2:" + str(delta2_nm_1) + "nm^-1",file=f)

    n2 = 3 * 2 * half_diver_ang * 0.001 / lam / delta2_nm_1
    n2 = math.ceil(n2)
    if (n2 % 2) == 0:
        pass
    else:
        n2 = n2 + 1
    
    print("n2:" + str(n2) + "pixels",file=f)

    m2 = 1.5 * n2
    print("m2:" + str(m2) + "pixels",file=f)

    d2 = w / m2
    print("d2:" + str(d2) + "nm",file=f)
    """
    f.close



    return