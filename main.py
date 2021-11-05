#########################################
#ANCHOR Python script for Master Thesis
#Determining Regularization and Simulating the Electron Beam in Non-translational Ptychography
#Guangyu He 
#########################################

#ANCHOR import python packages
import numpy as np
import math
import struct
import os
from matplotlib import pyplot as plt

#ANCHOR import self dependency
from bin import bin_pack,bin_unpack

#########################################

#SECTION mode parameters
tif_mode = False #STUB storing data in tiff image, default: False

image_mode = True #STUB presenting result in images, default: True

std_beam_mode = False #STUB replace Grillo's probe into standard beam, default: False

defocus_mode = True #STUB using defocus gaussian potential, default: True

loop_mode = False #STUB loop all 9 probes for creating rop data, default: False

if loop_mode:   #ANCHOR in loop mode, image mode is False by default
    image_mode = False 
    pass
else:   #ANCHOR if not in loop mode, one needs to specify mat probe and rotation

    #mat = '++++' #NOTE ++++ is not available in this thesis: bad interpolation value

    #mat = '-.-.-.+'
    #mat = '-.-.+.+'
    mat = '-.+.-.+'

    rotation = 0 #0/-1/-2/-3
#!SECTION

#SECTION pre simulation parameters
pre_sim_mode = True #STUB enable a pre-simulation to obtain values for interpolation, incl. a report of widened beam in txt, default: True
sim_mode = True #STUB enable a simulation to obtain beam information after interpolation, written in a txt report, default: True

if pre_sim_mode | sim_mode: #ANCHOR in either mode, self dependencies are required
    from simulation import pre_simulation,simulation
    from fwhm import fwhm
else:
    pass
#!SECTION

#SECTION value parameters
radius_mat = 100 #STUB determine the central disk of the probe, default: 100

#ANCHOR value for gaussian potential
atom_distance = 34 #STUB define atom distance for simulated object potential, default: 34. 36 is also a possible value
rescale_range = math.pi / 4 #STUB define the normalization factor of the total gaussian potential, different value may have effect in diffraction scattering, default: math.pi / 4

#ANCHOR force dimension and interpolation value in program
#NOTE two values are adjusted for all 9 probes
f_dimension = 144
f_dimension_interpolate = 32
#!SECTION

#SECTION main function
def main():
    """
    main function
    """

    #ANCHOR load .mat file(in freq. distri.)
    value_mat = loadmat('quad'+mat+'pix256.mat')

    #ANCHOR add a rotation
    value_mat = np.rot90(value_mat,rotation)

    #ANCHOR cut the beam from phase plate
    #STUB default center out of 256x256 is [128,128]
    value_mat_cut = value_mat[128-radius_mat:128+radius_mat,128-radius_mat:128+radius_mat]

    #SECTION pre_simulation block
    if pre_sim_mode:

        #ANCHOR add an edge to the beam in freq. distri.
        value_pre_edged = add_edge(value_mat_cut,200)

        #ANCHOR caluculate f=Aexp(i\phi) before fft, see book
        value_pre_f1 = np.exp((0+1j) * value_pre_edged)
        #ANCHOR define A as the place of original beam with 1 and outer with zero
        value_pre_f = circle(value_pre_f1,radius_mat)

        #ANCHOR ifft shift to center
        value_pre_f = np.fft.ifftshift(value_pre_f) 
        #ANCHOR ifft from rec. space to real space
        value_pre_ifft = np.fft.ifft2(value_pre_f)
        #ANCHOR shift to center
        value_pre_probe = np.fft.fftshift(value_pre_ifft)
    
        #ANCHOR absolute squared for Intensity
        value_pre_probe_tosimulation = np.abs(value_pre_probe)**2
        #ANCHOR calculate the fwhm of the beam
        fwhm_pre_probe_pixels = fwhm(value_pre_probe_tosimulation)

        #ANCHOR pre_simulation() returns interpolation values
        Dimension, delta1, delta2 = pre_simulation(value_pre_probe_tosimulation,radius_mat,fwhm_pre_probe_pixels,mat) #m2

        #SECTION adjustment for interpolation values
        if Dimension % 2 != 0:
            Dimension = Dimension + 1
        else:
            pass

        if mat=="-.+.-.+":
            fix_parameter = 12
        else:
            fix_parameter = 8

        Dim_interpolate = int(delta1 / delta2 * Dimension) + fix_parameter
        
        if Dim_interpolate % 2 != 0:
            Dim_interpolate = Dim_interpolate + 1
        else:
            pass
        #!SECTION
    else:
        pass
    #!SECTION

    #ANCHOR overwrite dim and interpolation value with pre-factor
    Dimension = f_dimension
    Dim_interpolate = f_dimension_interpolate

    #ANCHOR new radius of interpolated probe
    radius = Dim_interpolate / 2
    #ANCHOR how many zero pixels will be added in to the new cutted image
    edge = int(Dimension / 2 - radius)
    #ANCHOR linear interpolation to dimension of m
    value_interpolated = Bilinear(value_mat_cut,Dim_interpolate,Dim_interpolate)
    #ANCHOR add an edge to the beam in freq. distri.
    value_edged = add_edge(value_interpolated,edge)

    #ANCHOR caluculate f=Aexp(i\phi)
    value_f = np.exp((0+1j) * value_edged)
    #ANCHOR define A as the place of original beam with 1 and outer with zero
    value_f = circle(value_f,radius)

    #ANCHOR a hanning window to the dataset
    full_window = np.hanning(value_f.shape[0])[:,None]
    full_window = np.sqrt(np.dot(full_window, full_window.T))**2
    value_f *= full_window

    #ANCHOR ifft shift to center
    value_f = np.fft.ifftshift(value_f) 
    #ANCHOR ifft, from rec. space to real space
    value_ifft = np.fft.ifft2(value_f)
    #ANCHOR fftshift to center
    value_probe_edge = np.fft.fftshift(value_ifft)
    
    #SECTION simulate the beam
    if sim_mode:
        #ANCHOR calculate the fwhm of the beam
        value_probe_tosimulation = np.abs(value_probe_edge)**2
        fwhm_probe_pixels = fwhm(value_probe_tosimulation)

        #ANCHOR simulate the beam
        simulation(value_probe_tosimulation,radius,fwhm_probe_pixels,mat)
    else:
        pass
    #!SECTION

    #ANCHOR create an object potential: grid of gaussian atoms
    #NOTE gold atom is simulated
    potential = gaussV(int(Dimension),rescale_range)

    #SECTION apply a defocus mode to illuminate more atoms on specimen
    if defocus_mode:

        #ANCHOR defocus potential using image zoom
        #NOTE 6 times zoom in
        defocus_V = defocus_potential(potential)

        #SECTION showing defocus image when image mode is on
        if image_mode:
            from matplotlib_scalebar.scalebar import ScaleBar 
            from matplotlib_scalebar.scalebar import SI_LENGTH

            plt.subplot(121),plt.imshow(np.abs(potential),cmap='jet',interpolation='None',aspect='1'),plt.title('original grid gaussian atoms')
            scalebar1 = ScaleBar(0.0135,'nm', SI_LENGTH)
            plt.gca().add_artist(scalebar1) 
            plt.subplot(122),plt.imshow(np.abs(defocus_V),cmap='jet',interpolation='None',aspect='1'),plt.title('defocused grid gaussian atoms')
            scalebar2 = ScaleBar(0.0135 * 6,'nm', SI_LENGTH)
            plt.gca().add_artist(scalebar2)
            
            cax = plt.axes([0.85, 0.1, 0.025, 0.8])
            plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            plt.colorbar(cax=cax)
            plt.show()
        #!SECTION
            
        potential = defocus_V
    #!SECTION

    #ANCHOR normalizations(sum of each pixel to one) of the probe
    value_probe_scale = uni_complex(value_probe_edge)

    #REVIEW potential is not necessary to normalize: potential = uni_complex(potential)

    #SECTION replace with a standard beam
    if std_beam_mode:
        #ANCHOR a exactly same way of init a standard beam from ROP
        from test_psi import standardbeam
        value_probe_scale = standardbeam()
    else:
        pass
    #!SECTION

    #SECTION check probe validity
    data_sum = np.sum(np.real(value_probe_scale)**2 + np.imag(value_probe_scale)**2)
    print('sum of probe:'+str(data_sum))
    for i in range(Dimension):
        for j in range(Dimension):
            if (np.abs(value_probe_scale[i,j]) < 0):
                print(str(i)+','+str(j)+'is < 0')
                break
            if np.isnan(value_probe_scale[i,j]):
                print(str(i)+','+str(j)+'is NaN')
                break
    #!SECTION

    #ANCHOR store potential values in .bin
    bin_pack(potential,'Potential')

    #SECTION store probe values
    #SECTION in .tiff image format
    #NOTE since .bin problem is fixed, tif mode is no longer a recommend.
    if tif_mode:

        #ANCHOR separate complex probe values into real and imag part.
        value_probe_real = np.real(value_probe_scale)
        value_probe_imag = np.imag(value_probe_scale)

        #ANCHOR import tif denpendency and store value as tif image
        #NOTE a strength constant is used,e.g., 10000 here, when creating tif image, and using uint32 format as default.
        from PIL import Image
        imreal = Image.fromarray((value_probe_real * 10000).astype(np.uint32))
        imreal.save('bin/Probe_'+str(mat)+'_'+str(rotation)+'_re.tif')
        imimag = Image.fromarray((value_probe_imag * 10000).astype(np.uint32))
        imimag.save('bin/Probe_'+str(mat)+'_'+str(rotation)+'_im.tif')

        #ANCHOR to make sure tif images are saving correctly, as well as using them as probes in following processes.
        imread_real =  Image.open('bin/Probe_'+str(mat)+'_'+str(rotation)+'_re.tif')
        imarray_real = np.array(imread_real).astype(np.int32)
        imread_imag = Image.open('bin/Probe_'+str(mat)+'_'+str(rotation)+'_im.tif')
        imarray_imag = np.array(imread_imag).astype(np.int32)

        #ANCHOR recombine complex value and normalize value(the 10000 multiplicity above is now eliminated)
        imarray = imarray_real + imarray_imag * 1j
        imarray = uni_complex(imarray)
        #ANCHOR restore the probe value
        value_probe_scale = imarray

        try:
            #ANCHOR in tif mode, all probe saved as .bin will be deleted, additionally in loop mode, all tif image will be renamed for ROP usage.
            os.system('rm bin/Probe*.bin')

            os.system('cp bin/Probe_-.+.-.+_0_re.tif bin/Probe0_re.tif')
            os.system('cp bin/Probe_-.+.-.+_0_im.tif bin/Probe0_im.tif')

            os.system('cp bin/Probe_-.-.-.+_0_re.tif bin/Probe1_re.tif')
            os.system('cp bin/Probe_-.-.-.+_0_im.tif bin/Probe1_im.tif')
            os.system('cp bin/Probe_-.-.-.+_-1_re.tif bin/Probe2_re.tif')
            os.system('cp bin/Probe_-.-.-.+_-1_im.tif bin/Probe2_im.tif')
            os.system('cp bin/Probe_-.-.-.+_-2_re.tif bin/Probe3_re.tif')
            os.system('cp bin/Probe_-.-.-.+_-2_im.tif bin/Probe3_im.tif')
            os.system('cp bin/Probe_-.-.-.+_-3_re.tif bin/Probe4_re.tif')
            os.system('cp bin/Probe_-.-.-.+_-3_im.tif bin/Probe4_im.tif')

            os.system('cp bin/Probe_-.-.+.+_0_re.tif bin/Probe5_re.tif')
            os.system('cp bin/Probe_-.-.+.+_0_im.tif bin/Probe5_im.tif')
            os.system('cp bin/Probe_-.-.+.+_-1_re.tif bin/Probe6_re.tif')
            os.system('cp bin/Probe_-.-.+.+_-1_im.tif bin/Probe6_im.tif')
            os.system('cp bin/Probe_-.-.+.+_-2_re.tif bin/Probe7_re.tif')
            os.system('cp bin/Probe_-.-.+.+_-2_im.tif bin/Probe7_im.tif')
            os.system('cp bin/Probe_-.-.+.+_-3_re.tif bin/Probe8_re.tif')
            os.system('cp bin/Probe_-.-.+.+_-3_im.tif bin/Probe8_im.tif')
        except:
            pass
        #!SECTION
    
    #SECTION in .bin file format
    else:
        #ANCHOR convert all probes into individual .bin file.
        #NOTE if loop mode is disabled, only the selected probe will be stored, while rest of probes will using the stored bins. 
        bin_pack(value_probe_scale,'Probe_'+str(mat)+'_'+str(rotation))

        try:
            #ANCHOR in bin mode, all tif images are deleted
            os.system('rm -rf bin/*.tif')

            os.system('cp bin/Probe_-.+.-.+_0_re.bin bin/Probe0_re.bin')
            os.system('cp bin/Probe_-.+.-.+_0_im.bin bin/Probe0_im.bin')

            os.system('cp bin/Probe_-.-.-.+_0_re.bin bin/Probe1_re.bin')
            os.system('cp bin/Probe_-.-.-.+_0_im.bin bin/Probe1_im.bin')
            os.system('cp bin/Probe_-.-.-.+_-1_re.bin bin/Probe2_re.bin')
            os.system('cp bin/Probe_-.-.-.+_-1_im.bin bin/Probe2_im.bin')
            os.system('cp bin/Probe_-.-.-.+_-2_re.bin bin/Probe3_re.bin')
            os.system('cp bin/Probe_-.-.-.+_-2_im.bin bin/Probe3_im.bin')
            os.system('cp bin/Probe_-.-.-.+_-3_re.bin bin/Probe4_re.bin')
            os.system('cp bin/Probe_-.-.-.+_-3_im.bin bin/Probe4_im.bin')

            os.system('cp bin/Probe_-.-.+.+_0_re.bin bin/Probe5_re.bin')
            os.system('cp bin/Probe_-.-.+.+_0_im.bin bin/Probe5_im.bin')
            os.system('cp bin/Probe_-.-.+.+_-1_re.bin bin/Probe6_re.bin')
            os.system('cp bin/Probe_-.-.+.+_-1_im.bin bin/Probe6_im.bin')
            os.system('cp bin/Probe_-.-.+.+_-2_re.bin bin/Probe7_re.bin')
            os.system('cp bin/Probe_-.-.+.+_-2_im.bin bin/Probe7_im.bin')
            os.system('cp bin/Probe_-.-.+.+_-3_re.bin bin/Probe8_re.bin')
            os.system('cp bin/Probe_-.-.+.+_-3_im.bin bin/Probe8_im.bin')
        except:
            pass
    #!SECTION
#!SECTION

    #ANCHOR create diffaction pattern, using fft and fftshift, check book
    diffraction_full = np.abs( np.fft.fft2( np.exp((0+1j) * potential) * value_probe_scale ) )**2
    diffraction_full = np.fft.fftshift(diffraction_full)

    #ANCHOR remove wrap-around artifacts: idea is to zero the outside 2/3 place, and resize the pattern using 2/3(i.e. m144->n96) of dim.
    centroide = int( len(diffraction_full)/2 )
    radius_23 = int( len(diffraction_full)/3 )
    diffraction_23_full = diffraction_full[centroide-radius_23:centroide+radius_23,centroide-radius_23:centroide+radius_23] # n2
    diffraction_23 = wrap_artifact(diffraction_23_full)

    #ANCHOR normalize the pattern
    #NOTE pattern only includes real part, so here only the real part is normalized!
    diffraction_23 = uni_complex_real(diffraction_23)
    #ANCHOR check if the pattern is correctly normalized
    diff_sum = np.sum(np.real(diffraction_23))
    print('sum of diffraction:'+str(diff_sum))

    #ANCHOR store diffraction patterns into bin file
    bin_pack(diffraction_23,'Diffraction_'+str(mat)+'_'+str(rotation))

    #SECTION imaging mode
    if image_mode:
        plt.clf()
        plt.title('beam in Rec. Space')
        plt.subplot(231),plt.imshow(np.abs(value_mat),cmap='jet',interpolation='None',aspect='1')#,vmin=0.4, vmax=1) #
        cb = plt.colorbar()
        plt.title('electrode configuration '+mat)
        plt.subplot(232),plt.imshow(np.real(value_mat_cut),cmap='jet',interpolation='None',aspect='1')#,vmin=0.4, vmax=1) #
        plt.title('value_cut ' + str(radius_mat*2))
        cb = plt.colorbar()
        plt.subplot(233),plt.imshow(np.real(value_interpolated),cmap='jet',interpolation='None',aspect='1')#,vmin=0.4, vmax=1) #
        plt.title('value_interpolated ' + str(Dim_interpolate))
        cb = plt.colorbar()
        plt.subplot(234),plt.imshow(np.real(value_edged),cmap='jet',interpolation='None',aspect='1')#,vmin=0.4, vmax=1) #
        plt.title('value_edged ' + str(edge*2 + Dim_interpolate))
        cb = plt.colorbar()
        plt.subplot(235),plt.imshow(np.fft.fftshift(np.real(value_f)),cmap='jet',interpolation='None',aspect='1')#,vmin=0.4, vmax=1) #
        plt.title('value_f ' + str(edge*2 + Dim_interpolate))
        cb = plt.colorbar()

        plt.figure()
        plt.title('beam in Real Space, object potential')
        plt.subplot(131),plt.imshow(np.abs(value_probe_edge),cmap='jet',interpolation='None',aspect='1')#,vmin=0.4, vmax=1) #
        plt.title('probe ' + str(edge*2 + Dim_interpolate))

        cb = plt.colorbar()
        plt.subplot(132),plt.imshow(np.abs(value_probe_scale),cmap='jet',interpolation='None',aspect='1')#,vmin=0.4, vmax=1) #
        plt.title('probe_scale' + str(Dimension) )
        cb = plt.colorbar()
        scalebar1 = ScaleBar(0.0135,'nm', SI_LENGTH)
        plt.gca().add_artist(scalebar1) 
        plt.subplot(133),plt.imshow(np.abs(potential),cmap='jet',interpolation='None',aspect='1')#,vmin=0.4, vmax=1) #
        plt.title('potential ' + str(Dimension) )
        cb = plt.colorbar()

        plt.figure()
        plt.title('diffraction')
        plt.subplot(231),plt.imshow(np.log(diffraction_full),cmap='jet',interpolation='None',aspect='1')#,vmin=0.4, vmax=1) #
        plt.title('diffraction_23 log ' + str(Dimension))
        cb = plt.colorbar()
        plt.subplot(232),plt.imshow(np.log(diffraction_23_full),cmap='jet',interpolation='None',aspect='1')#,vmin=0.4, vmax=1) #
        plt.title('diffraction_23_full log ' + str(radius_23 * 2))
        cb = plt.colorbar()
        plt.subplot(233),plt.imshow(np.log(diffraction_23),cmap='jet',interpolation='None',aspect='1')#,vmin=0.4, vmax=1) #
        plt.title('diffraction_23_wrap log ' + str(radius_23 * 2))
        cb = plt.colorbar()
        plt.subplot(234),plt.imshow(np.real(diffraction_full),cmap='jet',interpolation='None',aspect='1')#,vmin=0.4, vmax=1) #
        plt.title('diffraction_full' + str(Dimension))
        cb = plt.colorbar()
        plt.subplot(235),plt.imshow(np.real(diffraction_23),cmap='jet',interpolation='None',aspect='1')#,vmin=0.4, vmax=1) #
        plt.title('diffraction_23_wrap ' + str(radius_23 * 2))
        cb = plt.colorbar()

        plt.show()
    else:
        pass
    #!SECTION

    #SECTION create probe bundle for ROP
    try:
        os.system('cp bin/Diffraction_-.+.-.+_0_re.bin bin/dps0.bin')
        os.system('cp bin/Diffraction_-.-.-.+_0_re.bin bin/dps1.bin')
        os.system('cp bin/Diffraction_-.-.-.+_-1_re.bin bin/dps2.bin')
        os.system('cp bin/Diffraction_-.-.-.+_-2_re.bin bin/dps3.bin')
        os.system('cp bin/Diffraction_-.-.-.+_-3_re.bin bin/dps4.bin')
        os.system('cp bin/Diffraction_-.-.+.+_0_re.bin bin/dps5.bin')
        os.system('cp bin/Diffraction_-.-.+.+_-1_re.bin bin/dps6.bin')
        os.system('cp bin/Diffraction_-.-.+.+_-2_re.bin bin/dps7.bin')
        os.system('cp bin/Diffraction_-.-.+.+_-3_re.bin bin/dps8.bin')
        dps_list = []
        for i in range(9):
            dps = bin_unpack("bin/dps"+str(i)+".bin")
            #dps = bin_unpack("bin/Diffraction_re.bin")
            dps_list.append(dps)

        dps_list_numpy = np.array(dps_list)

        data = np.ravel(dps_list_numpy)
        data = struct.pack(9 * 96 * 96 *'f', *data)
        f = open('bin/DPS_sum.bin', 'wb')
        f.write(data)
        f.close()
        os.system('rm -rf bin/dps*.bin')
        if tif_mode:
            os.system('rm -rf bin/*im.bin')
        else:
            pass
    except:
        pass
    #!SECTION

#SECTION function: remove wraparound artifact
def wrap_artifact(value):
    """
    removing wraparound artifact, returns the central disk with 2/3 of dim.
    """
    radius = int( len(value)/2 )
    value_wrap = circle(value,radius)
    return value_wrap
#!SECTION

#SECTION functions: data normalization
def uni_complex(data):
    """
    returns normalized complex data
    """
    data /= np.sqrt( np.sum( np.real(data) * np.real(data)) + np.sum( np.imag(data) * np.imag(data)) ) 
    return data

def uni_complex_real(data):
    """
    returns normalized real data
    """
    print(np.sum(data))
    data /= np.sum(data)
    return data
#!SECTION

#SECTION function: create gaussian atoms
def gaussV(Dimension,rescale_range):
    """
    creates and returns the object potential with a grid gauss atom
    """

    #ANCHOR Create grid with ones
    ones = np.zeros((Dimension,Dimension))
    ones[::atom_distance, ::atom_distance] = 1.0

    #ANCHOR Create gaussian shaped atom
    sigma = 6
    center = int(Dimension / 2)
    gaus = np.zeros_like(ones)
    for x in range(Dimension):
        for y in range(Dimension):
            gaus[x,y] = (1/(2*np.pi*sigma**2)) * np.exp( - (1 / 2) * ( ( (x - center) / sigma)**2 + ( ( y - center) / sigma )**2 ) )
            #STUB test function: gaus[x,y] = 0

    #ANCHOR convolve operation using FFT
    #REVIEW probably not correct, since it didn't test since the scipy method is used
    """
    f_ones = np.fft.fft2(ones)
    f_gaus = np.fft.fft2(gaus)
    ifft( fft(array1) x fft(array2) )
    combined = np.fft.ifft2(f_ones * f_gaus)
    """

    #ANCHOR convolve operation using scipy dependency
    from scipy import signal
    combined = signal.fftconvolve(ones, gaus, mode='same')

    #ANCHOR showing the process of convolving
    """
    if image_mode:
        plt.figure()
        plt.subplot(1,3,1),plt.imshow(np.abs(ones),cmap='jet',interpolation='None',aspect='1')
        plt.subplot(1,3,2),plt.imshow(np.abs(gaus),cmap='jet',interpolation='None',aspect='1')
        plt.subplot(1,3,3),plt.imshow(np.abs(combined),cmap='jet',interpolation='None',aspect='1')
        plt.show()
    else:
        pass
    """

    #ANCHOR unification of potential, from 0 to rescale_range
    combined = unification_interval(combined,0,rescale_range)
    #STUB test function: ones = np.ones((Dimension,Dimension))

    return combined
#!SECTION

#SECTION function: load .mat files
def loadmat(mat):
    """
    load the .mat file, returns a numpy array with unification to [-pi,pi]
    """
    import scipy.io as sio

    #ANCHOR convert the image data from mat to numpy
    data = sio.loadmat(mat)
    value = np.transpose(data['f']) 

    #ANCHOR rescale between -pi to pi
    value = unification_interval(value,-math.pi,math.pi) 

    return value
#!SECTION

#SECTION function: unification
def unification_interval(data,interval_min,interval_max):
    """
    returns a rescaled  data
    """

    # data         ：需要变换的数据或矩阵
    # interval_min ：变换区间下限。
    # interval_max ：变换区间上限。
    data = np.array(data)
    n,m = data.shape
    minval = np.min(np.min(data))
    maxval = np.max(np.max(data))
    for i in range(n):
        for j in range(m):
            data[i,j] = (data[i,j]-minval)/(maxval-minval)
    return data*(interval_max-interval_min)+interval_min
#!SECTION

#SECTION function: extend the image with zeros
def add_edge(value,width):
    """
    returns extending dimension of image with zeros, value: old image, width: extending dim on one side
    """

    value_size = np.size(value,0)

    back0 = np.zeros((width,value_size)) #upper and down
    new = np.append(back0,value,axis=0) #width+size,size
    new = np.append(new,back0,axis=0) #width+size+width,size

    back1 = np.zeros((value_size+ 2*width, width)) #left and right
    new = np.append(back1,new,axis=1) # new_size,width+size
    new = np.append(new,back1,axis=1) # new_size,width+size+width

    #ANCHOR showing the result of widened image
    """
    if image_mode:
        plot_image(new,'widened image')
    else:
        pass
    """

    return new
#!SECTION

#SECTION function: cut out central disk
def circle(value,radius):
    """
    returns a image with a radius central disk cutted out
    """

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
#!SECTION

#SECTION function: cut out central disk with central disk equals one
def circle_one(value,radius):
    """
    returns a image with a radius central disk with value one cutted out
    """

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
    
    return a
#!SECTION

#SECTION function: linear interpolation
def Bilinear( img, bigger_height, bigger_width):
    """
    returns blinear interpolation
    """

    bilinear_img = np.zeros( shape = ( bigger_height, bigger_width))
    
    for i in range( 0, bigger_height ):
        for j in range( 0, bigger_width ):
            row = ( i / bigger_height ) * img.shape[0]
            col = ( j / bigger_width ) * img.shape[1]
            row_int = int( row )
            col_int = int( col )
            u = row - row_int
            v = col - col_int
            if row_int == img.shape[0]-1 or col_int == img.shape[1]-1:
                row_int -= 1
                col_int -= 1
                
            bilinear_img[i][j] = (1-u)*(1-v) *img[row_int][col_int] + (1-u)*v*img[row_int][col_int+1] + u*(1-v)*img[row_int+1][col_int] + u*v*img[row_int+1][col_int+1]
            
    return bilinear_img
#!SECTION

#SECTION function: defocus potential using zoom
def defocus_potential(value):
    from scipy import ndimage, misc

    result = ndimage.zoom(value, 1/6)
    result = add_edge(result,60)

    return result
#!SECTION

#SECTION program entrance
if __name__ == "__main__":
    if loop_mode:
        mat_list = ['-.-.-.+','-.-.+.+']
        mat = '-.+.-.+'
        rotation = 0
        main()
        for i in range(2):
            for j in range(4):
                mat = mat_list[i]
                if j == 0:
                    rotation = 0
                else:
                    rotation = -j
                main()
    else:
        main()
#!SECTION