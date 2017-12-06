import tensorflow as tf
from scipy import misc
import matplotlib.pyplot as plt


IN_PATH_A = "images/blurA1.jpg"
IN_PATH_A_P = "images/rose-src.jpg" 
IN_PATH_B = "images/blurB1.jpg"
OUT_PATH_B_P = "images/out.jpg"	 


def 
def read_image(image_path):
	im = misc.imread(image_path)
	# plt.imshow(im)
	# plt.show()
	return im

def main():
	A = read_image(IN_PATH_A)
	A_P = read_image(IN_PATH_A_P)
	B = read_image(IN_PATH_B)

if __name__ == '__main__':
   main()

