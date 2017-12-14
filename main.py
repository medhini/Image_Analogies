from scipy import misc
import colorsys
import matplotlib.pyplot as plt
import numpy as np
from gaussian_pyramid import compute_gaussian_pyramid
from features import compute_features, concat_features, extract_pixel_feature
from approximate_match import ann_index, best_approximate_match
from coherence_match import best_coherence_match
from utils import *
from numpy.linalg import norm

IN_PATH_A = "images/blurA1.jpg"
IN_PATH_A_P = "images/rose-src.jpg"
IN_PATH_B = "images/blurB1.jpg"
OUT_PATH_B_P = "images/out.jpg"
LUM_REMAP = False

max_level = 5  # set the max level of pyramid
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


	"""Remap luminance for color artistic images"""
	if LUM_REMAP:
		im_a, im_a_p = remap_luminance(im_a, im_a_p, im_b)

	pyramid_a = compute_gaussian_pyramid(im_a, max_level)
	pyramid_a_p = compute_gaussian_pyramid(im_a_p, max_level)
	pyramid_b = compute_gaussian_pyramid(im_b, max_level)
	pyramid_b_p = pyramid_b
	im_b_yiq = colorsys.rgb_to_yiq(im_b[:,:,0],im_b[:,:,1],im_b[:,:,2])
	pyramid_color = compute_gaussian_pyramid(im_b_yiq, max_level)

	
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

		for row in range(imh):
			for col in range(imw):
				px = np.array([row, col])

				# do something about B and Bp feature
				feature_b_p = compute_features(pyramid_b_p)
				small_padded = np.pad(feature_b_p[level-1], (n_sm//2), 'reflect') 
				big_padded = np.pad(feature_b_p[level], (n_lg//2), 'reflect')
				BBp_feature = np.hstack([features_b[level][to_1d(px, imw), :],
									  extract_pixel_feature(small_padded, big_padded, px)])

				assert(BBp_feature.shape == (As_size[level][1],))
				# Find Approx Nearest Neighbor
				p_app_ix = best_approximate_match(flann[level], flann_params[level], BBp_feature)

				Ap_imh, Ap_imw = pyramid_a_p[level].shape[:2]
				p_app = to_2d(p_app_ix, Ap_imw)


				if(len(s)<1):
					p = p_app

				else:
					#Coherence match   
					p_coh = best_coherence_match(As[level], (Ap_imh, Ap_imw), BBp_feature, s, px, imw, n_lg)

					if np.allclose(p_coh, np.array([-1, -1])):
						p = p_app

					else:
						AAp_feature_app = As[level][p_app]
						AAp_feature_coh = As[level][p_coh]
						d_app = norm(AAp_feature_app - BBp_feature)**2
						d_coh = norm(AAp_feature_coh - BBp_feature)**2

						if d_coh < d_app * (1 + (2**(level-5)*1)):
							p = p_coh
						else:
							p = p_app

				# print len(tuple(p))
					# print p_coh
				# print pyramid_a_p[level][tuple(p)].shape, pyramid_b_p[level][row, col].shape
				s.append(p)
				pyramid_b_p[level][row, col] = pyramid_a_p[level][tuple(p)]

		
		# Save color output images	
		im_out_yiq = np.dstack([pyramid_b_p[level], pyramid_color[level][:, :, 1:]])
		color_im_out = colorsys.yiq_to_rgb(im_out_yiq[:,:,0], im_out_yiq[:,:,1], im_out_yiq[:,:,2])
		color_im_out = np.clip(color_im_out, 0, 1)

		plt.imsave('output/level_%d_color.jpg' % level, color_im_out)



if __name__ == '__main__':
   main()
