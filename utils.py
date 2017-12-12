from skimage.transform import pyramid_gaussian
import matplotlib.pyplot as plt

def compute_gaussian_pyramid(img, level = 5):
	"""Computes a gaussian pyramid for an input image"""

	image_pyramid = list(pyramid_gaussian(img, max_layer = level))
	image_pyramid.reverse()

	return image_pyramid


def remap_luminance(im_a, im_a_p, im_b):
    # compute luminance
    YIQ_tmp = colorsys.rgb_to_yiq(im_a[:,:,0],im_a[:,:,1],im_a[:,:,2])
    lum_img_a = YIQ_tmp[0]
    YIQ_tmp = colorsys.rgb_to_yiq(im_a_p[:,:,0],im_a_p[:,:,1],im_a_p[:,:,2])
    lum_img_a_p = YIQ_tmp[0]
    YIQ_tmp = colorsys.rgb_to_yiq(im_b[:,:,0],im_b[:,:,1],im_b[:,:,2])
    lum_img_b = YIQ_tmp[0]

	mean_a = np.mean(lum_img_a)
    mean_b = np.mean(lum_img_b)
    std_dev_a = np.std(lum_img_a)
    std_dev_b = np.std(lum_img_b)

    img_a_remap = (std_dev_b/std_dev_a) * (lum_img_a - mean_a) + mean_b

    img_a_p_remap = []

    for im in lum_img_a_p:
        img_a_p_remap.append((std_dev_b/std_dev_a) * (im - mean_a) + mean_b)

    return img_a_remap, img_a_p_remap