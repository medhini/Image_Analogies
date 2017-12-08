from scipy import misc
import matplotlib.pyplot as plt
from gaussian_pyramid import compute_gaussian_pyramid
from features import compute_features
from features import concat_features

IN_PATH_A = "images/blurA1.jpg"
IN_PATH_A_P = "images/rose-src.jpg"
IN_PATH_B = "images/blurB1.jpg"
OUT_PATH_B_P = "images/out.jpg"

def read_image(image_path):
	im = misc.imread(image_path)
	# plt.imshow(im)
	# plt.show()
	return im

def main():
	im_a = read_image(IN_PATH_A)
	im_a_p = read_image(IN_PATH_A_P)
	im_b = read_image(IN_PATH_B)

	image_pyramid_a = compute_gaussian_pyramid(im_a)
	image_pyramid_a_p = compute_gaussian_pyramid(im_a_p)
	image_pyramid_b = compute_gaussian_pyramid(im_b)

	# Test to show how to retrieve concatenated features
	features = concat_features(image_pyramid_a)
	for i in range(1,len(features)):
		print(features[i].shape)


	"""Computing Luminances"""

	# lum_a, lum_a_pyramid, lum_b
	# """Remap luminance for color artistic images"""
	# im_a, image_pyramid_a = remap_luminance(lum_a, lum_a_pyramid, lum_b)

	""""""


if __name__ == '__main__':
   main()
