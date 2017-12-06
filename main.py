from scipy import misc
import matplotlib.pyplot as plt
from gaussian_pyramid import compute_gaussian_pyramid
from features import compute_features

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
	# im_a_p = read_image(IN_PATH_A_P)
	# im_b = read_image(IN_PATH_B)

	image_pyramid_a = compute_gaussian_pyramid(im_a)
	# image_pyramid_a_p = compute_gaussian_pyramid(im_a_p)
	# image_pyramid_b = compute_gaussian_pyramid(im_b)
	features = compute_features(image_pyramid_a)
	print(features)

if __name__ == '__main__':
   main()
