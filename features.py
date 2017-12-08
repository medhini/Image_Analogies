import colorsys
import numpy as np

def concat_features(pyramid):
    features = compute_features(pyramid)

    F = [[]]
    small_window = 3
    big_window = 5
    small_pad = small_window//2
    big_pad = big_window//2

    for i in range(1,len(pyramid)):
        fl = np.zeros(features[i].shape+(small_window**2+big_window**2,))
        small_padded = np.pad(features[i-1], (small_pad,), 'reflect')
        big_padded = np.pad(features[i], (big_pad,), 'reflect')

        for y in range(features[i].shape[0]):
            for x in range(features[i].shape[1]):
                small_x = x//2 + small_pad
                small_y = y//2 + small_pad
                big_x = x + big_pad
                big_y = y + big_pad

                small_patch = small_padded[small_y-small_pad:small_y+small_pad+1, small_x-small_pad:small_x+small_pad+1]
                big_patch = big_padded[big_y-big_pad:big_y+big_pad+1, big_x-big_pad:big_x+big_pad+1]

                small_vector = small_patch.flatten()
                big_vector = big_patch.flatten()
                fl[y,x,:] = np.hstack((big_vector, small_vector))
        F.append(fl)

    return F


def compute_features(pyramid):
    features = []
    for i in range(len(pyramid)):
        image = pyramid[i]
        YIQ = colorsys.rgb_to_yiq(image[:,:,0],image[:,:,1],image[:,:,2])
        features.append(YIQ[0])
    return features
