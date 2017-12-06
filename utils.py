from skimage.transform import pyramid_gaussian
import matplotlib.pyplot as plt

def compute_gaussian_pyramid(img, level = 5):
	"""Computes a gaussian pyramid for an input image"""

	image_pyramid = list(pyramid_gaussian(img, max_layer = level))
	image_pyramid.reverse()

	return image_pyramid


def remap_luminance(lum_img_a, lum_img_a_p, lum_img_b):

	mean_a = np.mean(lum_img_a)
    mean_b = np.mean(lum_img_b)
    std_dev_a = np.std(lum_img_a)
    std_dev_b = np.std(lum_img_b)

    img_a_remap = (std_dev_b/std_dev_a) * (lum_img_a - mean_a) + mean_b

    img_a_p_remap = []

    for im in lum_img_a_p:
        img_a_p_remap.append((std_dev_b/std_dev_a) * (im - mean_a) + mean_b)

    return img_a_remap, img_a_p_remap