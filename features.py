import colorsys

def compute_features(pyramid):
    features = []
    for i in range(len(pyramid)):
        image = pyramid[i]
        YIQ = colorsys.rgb_to_yiq(image[:,:,0],image[:,:,1],image[:,:,2])
        features.append(YIQ[0])
    return features
