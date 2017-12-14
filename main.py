from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from gaussian_pyramid import compute_gaussian_pyramid
from features import compute_features, concat_features, extract_pixel_feature
from approximate_match import ann_index, best_approximate_match
from coherence_match import best_coherence_match
from utils import *

IN_PATH_A = "images/blurA1.jpg"
IN_PATH_A_P = "images/rose-src.jpg"
IN_PATH_B = "images/blurB1.jpg"
OUT_PATH_B_P = "images/out.jpg"
LUM_REMAP = False

AB_weight = 1  # relative weighting of A and B relative to Ap and Bp
k    = 0.5     # 0.5 <= k <= 5 for texture synthesis
n_sm = 3       # coarse scale neighborhood size
n_lg = 5       # fine scale neighborhood size

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

	"""Computing Luminances"""

	lum_a, lum_a_pyramid, lum_b = rgb_to_yiq()
	"""Remap luminance for color artistic images"""
	im_a, im_a_p = remap_luminance(im_a, im_a_p, im_b)
	im_a, pyramid_a_p = remap_luminance(lum_a, lum_a_pyramid, lum_b)

	pyramid_a = compute_gaussian_pyramid(im_a)
	pyramid_a_p = compute_gaussian_pyramid(im_a_p)
	pyramid_b = compute_gaussian_pyramid(im_b)
	pyramid_b_p = pyramid_b

	
	# Compute features of B
	features_b = concat_features(pyramid_b)

	# Build structure for ANN
	flann, flann_params, As, As_size = ann_index(pyramid_a, pyramid_a_p)

	##################################################################
	# Algorithms

	for level in range(1, len(pyramid_a)):
		print('Computing level %d of %d' % (level, len(pyramid_a)-1))

		imh, imw = pyramid_b[level].shape[:2]
		im_out = np.nan * np.ones((imh, imw, 3))

		s = []
		im = []

		for row in range(imh):
			for col in range(imw):
				px = np.array([row, col])

				# do something about B and Bp feature
				feature_b_p = compute_features(pyramid_b_p)
				small_padded = np.pad(feature_b_p[level-1], (3//2), 'reflect') 
				big_padded = np.pad(feature_b_p[level], (5//2), 'reflect')
				BBp_feature = np.hstack([features_b[level][to_1d(px, imw), :],
									  extract_pixel_feature(small_padded, big_padded, px)])

				assert(BBp_feature.shape == (As_size[level][1],))
				
				# Find Approx Nearest Neighbor
				p_app_ix = best_approximate_match(flann[level], flann_params[level], BBp_feature)


				Ap_imh, Ap_imw = im_a_p.shape[:2]
				p_app, i_app = Ap_to_2d(p_app_ix, Ap_imh, Ap_imw)

				#Coherence match

				if(len(s)<1):
					p = p_app
					i = i_app

				else:   
					p_coh, i_coh, r_star = best_coherence_match(image_pyramid_a[level], (Ap_imh, Ap_imw), BBp_feature, s, im, px, imw, n_lg)

					if np.allclose(p_coh, np.array([-1, -1])):
						p = p_app
						i = i_app

					else:
						AAp_feature_app = As[level][p_app]
						AAp_feature_coh = As[level][p_coh]
						d_app = norm(AAp_feature_app - BBp_feature)**2
						d_coh = norm(AAp_feature_coh - BBp_feature)**2

						if d_coh < d_app * (1 + (2**(level-5)*1)):
							p = p_coh
							i = i_coh
						else:
							p = p_app
							i = i_app


				# pyramid_b_p[level][row, col] = pyramid_a_p[i][level][tuple(p)]

				# s.append(p)


	
	""""""


if __name__ == '__main__':
   main()
