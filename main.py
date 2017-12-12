from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from gaussian_pyramid import compute_gaussian_pyramid
from features import compute_features, concat_features, extract_pixel_feature
from approximate_match import ann_index, best_approximate_match, px2idx, idx2px

IN_PATH_A = "images/blurA1.jpg"
IN_PATH_A_P = "images/rose-src.jpg"
IN_PATH_B = "images/blurB1.jpg"
OUT_PATH_B_P = "images/out.jpg"
LUM_REMAP = False

def main():
	im_a = misc.imread(IN_PATH_A)
	im_a_p = misc.imread(IN_PATH_A_P)
	im_b = misc.imread(IN_PATH_B)

	# image single to double (float in 0 to 1 scale)
	if np.max(im_a) > 1.0:
		im_a = im_a/255.
	if np.max(im_b) > 1.0:
		im_b = im_b/255.
	if np.max(im_a_p) > 1.0:
		im_a_p = im_a_p/255.

	pyramid_a = compute_gaussian_pyramid(im_a)
	pyramid_a_p = compute_gaussian_pyramid(im_a_p)
	pyramid_b = compute_gaussian_pyramid(im_b)

	# Compute features of B
	features_b = concat_features(pyramid_b)

	# Build structure for ANN
	flann, flann_params, s, s_size = ann_index(pyramid_a, pyramid_a_p)

	##################################################################
	# Algorithms

	for level in range(1, len(pyramid_a)):
		print('Computing level %d of %d' % (level, len(pyramid_a)-1))

		imh, imw = pyramid_b[level].shape[:2]
		im_out = np.nan * np.ones((imh, imw, 3))

		for row in range(imh):
			for col in range(imw):
				px = np.array([row, col])

				# do something about B and Bp feature
				small_padded = np.pad(pyramid_b[level-1], (3//2), 'reflect') 
				big_padded = np.pad(pyramid_b[level], (5//2), 'reflect')
				BBp_feature = np.hstack([features_b[level][px2idx(px, imw), :],
                                      extract_pixel_feature(small_padded, big_padded, px)])

				assert(BBp_feature.shape == (s_size[level][1],))
				# Find Approx Nearest Neighbor
				p_app_idx = best_approximate_match(flann[level], flann_params[level], BBp_feature)

				Ap_h, Ap_w = im_a_p.shape[:2]
				p_app = idx2px(p_app_idx, Ap_w)



	"""Computing Luminances"""

	# lum_a, lum_a_pyramid, lum_b
	# """Remap luminance for color artistic images"""
	# im_a, im_a_p = remap_luminance(im_a, im_a_p, im_b)
	# im_a, image_pyramid_a = remap_luminance(lum_a, lum_a_pyramid, lum_b)

	""""""


if __name__ == '__main__':
   main()
