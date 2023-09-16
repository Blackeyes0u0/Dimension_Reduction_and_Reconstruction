import struct

import numpy as np

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


if __name__ == '__main__':
    fname = 'train-images-idx3-ubyte'
    images = read_idx(fname)
    
    fname = 'train-labels-idx1-ubyte'
    labels = read_idx(fname)


    np.save('train-images.npy', images)
    np.save('train-labels.npy', labels)