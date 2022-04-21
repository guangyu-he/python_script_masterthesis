#########################################
#python script for Master Thesis
#Determining Regularization and Simulating the Electron Beam in Non-translational Ptychography
#Guangyu He
#########################################

import numpy as np
import struct

def bin_pack(value,type):
    """
    type should be a string declaim the type of the value(e.g. probe,potential) \n
    pack value into a bin file
    """

    value_real = np.ravel(value.real)
    data_real = struct.pack(value.shape[0] * value.shape[0] * 'f', *value_real)
    d = open('bin/'+type+'_re.bin', 'wb')
    d.write(data_real)
    d.close()

    value_imag = np.ravel(value.imag)
    data_imag = struct.pack(value.shape[0] * value.shape[0] * 'f', *value_imag)
    d = open('bin/'+type+'_im.bin', 'wb')
    d.write(data_imag)
    d.close()

def bin_unpack(path):
    """
    unpack value from a bin file
    """
    
    import math

    f = open(path, 'rb')
    data = f.read()
    f.close
    data_len = len(data)
    value = struct.unpack(('%df' % (data_len / 4)), data)
    Dimension = int(math.sqrt(data_len/4))
    #print('bin-Dim:' + str(Dimension))
    values=np.asarray(value)
    values=np.reshape(values,(Dimension,Dimension))

    return values
