import numpy as np

class TensorTransform:
    # The idea of this class is to take in different raw data and turn them into tensors of desired shape
    # There's got to be a better way to do this
    def transform(self, raw_data, scale_params, noise_scale):
        # I assume that the time dimension is the first that is passed
        shape = [len(a[:,0]) for a in raw_data]
        print(f"This is the shape? {shape}")
        Y = noise_scale*np.random.normal(0, 1, size = shape)
        outer = lambda x,y : np.multiply.outer(x,y) 
        if len(shape) == 2:
            for i in range(len(raw_data[0])):
                Y += scale_params[i]*outer(raw_data[0][:,i], raw_data[1][:,i])
        elif len(shape) == 3:
            print(f"This is the number of factors? {len(raw_data[0][0,:])}")
            for i in range(len(raw_data[0][0,:])):
                    Y += scale_params[i]*outer(outer(raw_data[0][:,i], raw_data[1][:,i]), raw_data[2][:,i]) 
        return Y