from skimage.transform import pyramid_gaussian
import matplotlib.pyplot as plt

def compute_gaussian_pyramid(img, level = 5):
	"""Computes a gaussian pyramid for an input image"""

	image_pyramid = list(pyramid_gaussian(img, max_layer = level))
	image_pyramid.reverse()

	return image_pyramid
