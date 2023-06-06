import imageio
import numpy as np


def rmse(my_image, high_img):
	if my_image.shape != high_img.shape:
		raise ValueError("Nao sao do mesmo tamanho")
	
	N = my_image.shape[0]
	diff = (my_image.astype(np.int32) - high_img.astype(np.int32)) ** 2
	mean = np.mean(diff)
	rmse = np.sqrt(mean)

	return rmse

def joint_histogram(image1, image2, image3, image4, channels=256):
	histogram1 = histogram(image1, 256)
	histogram2 = histogram(image2, 256)
	histogram3 = histogram(image3, 256)
	histogram4 = histogram(image4, 256)

	#sum all histograms together and divide by 4
	jch = (histogram1 + histogram2 + histogram3 + histogram4)/4
	return jch

def gamma_correction(image, gamma):
	#normalize image between 0 and 1
	image = image.astype('float32')/255.0
	#apllying gamma correction
	corrected_image = np.power(image, 1.0/gamma)
	#renomalizing image
	corrected_image = (corrected_image * 255.0).astype('uint8')

	return corrected_image
	
def superresolution(image1, image2, image3, image4):
	N, M = image1.shape
	#create empty image with 2x the size of others
	sup = np.empty((2*N, 2*M), dtype=np.uint8)
	# intercaleave the pixels of all images
	sup[::2, ::2] = image1	
	sup[::2, 1::2] = image2
	sup[1::2, ::2] = image3
	sup[1::2, 1::2] = image4

	return sup

def histogram(image, levels):
	N,M = image.shape
	hist = np.zeros(levels).astype(int)
	for i in range(levels):
		nopix_value_i = np.sum(image == i)
		hist[i] = nopix_value_i

	return hist

def histogram_equalization(image, levels, hist):
	#hist = histogram(image, 256)

	histC = np.zeros(levels).astype(int)
	histC[0] = hist[0]

	for i in range(1, levels):
		histC[i] = hist[i] + histC[i-1]

	hist_transf = np.zeros(levels).astype(np.uint8)
	N,M = image.shape
	image_eq = np.zeros([N, M]).astype(np.uint8)

	for z in range(levels):
		s = ((levels -1 )/float(M*N)) * histC[z]
		hist_transf[z] = s
		image_eq[np.where(image == z)] = s

	
	return image_eq


def main():
	#reading input of low resolution ones
	low_name = input()
	img1 = imageio.imread(low_name+"0.png")
	img2 = imageio.imread(low_name+"1.png")
	img3 = imageio.imread(low_name+"2.png")
	img4 = imageio.imread(low_name+"3.png")

	high_name = input()
	h_image = imageio.imread(high_name)
	
	#choose enhancement method (0 is no enhancement)
	enhancement = int(input())
	#read gamma param
	gamma_param = float(input())

	if(enhancement == 0):
		my_image = superresolution(img1, img2, img3, img4)
		res = rmse(my_image, h_image)
		

	if(enhancement == 1):
		#do histogram equalization with all separetadely
		hist = histogram(img1, 256)
		eq1  = histogram_equalization(img1, 256, hist)
		hist = histogram(img2, 256)
		eq2  = histogram_equalization(img2, 256, hist)
		hist = histogram(img3, 256)
		eq3 = histogram_equalization(img3, 256, hist)
		hist = histogram(img4, 256)
		eq4 = histogram_equalization(img4, 256, hist)

		my_image = superresolution(eq1, eq2, eq3, eq4)	
		res = rmse(my_image, h_image)
		
	if(enhancement == 2):
		hist = joint_histogram(img1, img2, img3, img4)
		eq1 = histogram_equalization(img1, 256, hist)
		eq2 = histogram_equalization(img2, 256, hist)
		eq3 = histogram_equalization(img3, 256, hist)
		eq4 = histogram_equalization(img4, 256, hist)

		my_image = superresolution(eq1, eq2, eq3, eq4)	
		res = rmse(my_image, h_image)

	if(enhancement == 3):
		corrected1 = gamma_correction(img1, gamma=gamma_param)
		corrected2 = gamma_correction(img2, gamma=gamma_param)
		corrected3 = gamma_correction(img3, gamma=gamma_param)
		corrected4 = gamma_correction(img4, gamma=gamma_param)
		my_image = superresolution(corrected1, corrected2, corrected3, corrected4)
		res = rmse(my_image, h_image)
		
	print('%.4f' % res)


if __name__ == '__main__':
	main()


