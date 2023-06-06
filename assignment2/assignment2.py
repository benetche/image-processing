import imageio.v3 as imageio
import numpy as np


def rmse(my_image, high_img):
	if my_image.shape != high_img.shape:
		raise ValueError("Nao sao do mesmo tamanho")
	
	diff = (my_image.astype(np.int32) - high_img.astype(np.int32)) ** 2
	mean = np.mean(diff)
	rmse = np.sqrt(mean)

	return rmse

def fourier(image):
	F = np.fft.fft2(image)
	Fshift = np.fft.fftshift(F)

	return Fshift

def reverseFourier(shifted):
	G = np.fft.ifftshift(shifted)
	g = np.fft.ifft2(G).real

	norm = 255*((g - g.min())/(g.max() - g.min()))

	return norm

def applyFilter(Fshift, filter):
	Gshift = Gshift = np.multiply(Fshift , filter)
	g = reverseFourier(Gshift)
	return g.astype(np.uint8)

def idealHighPass(Fshift, D0):
	#create ideal low-pass filter
	M,N = Fshift.shape
	filter = np.zeros((M,N) , dtype=np.float32)
	for u in range(M):
		for v in range(N):
			D = np.sqrt((u-M/2)**2 + (v-N/2)**2 )
			if D > D0:
				filter[u, v] = 1

	return filter

def idealLowPass(Fshift, D0):
	#create ideal low-pass filter
	M,N = Fshift.shape
	filter = np.zeros((M,N) , dtype=np.float32)
	for u in range(M):
		for v in range(N):
			D = np.sqrt((u-M/2)**2 + (v-N/2)**2 )
			if D <= D0:
				filter[u, v] = 1
			
	return filter

def idealBandpass( filter1, filter2):
	#subtract higher radius low-pass filter with lower radius low-pass filter
	bpfilter = np.abs(filter1 - filter2)

	return bpfilter

def laplacianHighPass(Fshift):
	M, N = Fshift.shape
	
	H = np.zeros((M, N), dtype=np.float32)
	for u in range(M) :
		for v in range(N):
			# High pass = 1 - lowpass 
			H[u, v] = 1 - ((u-M/2)**2 + (v-N/2)**2) * (-4 * (np.pi)**2 )

	return H

def gaussianLowPass(Fshift, dev1, dev2):
	M, N = Fshift.shape

	H = np.zeros((M, N), dtype=np.float32)
	for u in range(M) :
		x1 = ((u - M/2) ** 2) / (2 * (dev1 ** 2))
		for v in range(N):
			x = x1 + ((v - N/2) ** 2) / (2 * (dev2 ** 2)) 
			H[u,v] = np.exp(-x)

	return H

def butterWorthLowpass(Fshift, D0, n ):
	M, N = Fshift.shape
	H = np.zeros((M, N) , dtype=np.float32)

	for u in range(M):
		for v in range(N):
			D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
			H[u, v] = 1 / (1 + (D/D0)**(2*n))
		
	return H

def butterWorthHighpass(Fshift, D0, n ):
	M, N = Fshift.shape
	H = np.zeros((M, N) , dtype=np.float32)

	for u in range(M):
		for v in range(N):
			D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
			# Highpass is 1 - lowpass
			H[u, v] = 1 - (1 / (1 + (D/D0)**(2*n)))
		
	return H

filter_selector = [idealLowPass, idealHighPass, ]

def main():
	image_name = input().replace('\r', '')
	expected_name = input().replace('\r', '')

	input_img = imageio.imread(image_name)
	expected_img = imageio.imread(expected_name)

	cmnd = int(input())
	if cmnd == 0:
		radius = float(input())
		Fshift = fourier(input_img)
		filter = idealLowPass(Fshift, radius)
		g = applyFilter(Fshift, filter)

	if cmnd == 1:
		radius = float(input())
		Fshift = fourier(input_img)
		filter = idealHighPass(Fshift, radius)
		g = applyFilter(Fshift, filter)

	if cmnd == 2:
		r1 = float(input())
		r2 = float(input())
		Fshift = fourier(input_img)
		higher = idealHighPass(Fshift, r1)
		lower = idealHighPass(Fshift, r2)
		band = idealBandpass(filter1=higher, filter2=lower)
		g = applyFilter(Fshift, band)

	if cmnd == 3:
		Fshift = fourier(input_img)
		filter = laplacianHighPass(Fshift)
		g = applyFilter(Fshift, filter)

	if cmnd == 4:
		dev1 = float(input())
		dev2 = float(input())
		Fshift = fourier(input_img)
		filter = gaussianLowPass(Fshift, dev1, dev2)
		g = applyFilter(Fshift, filter)


	if cmnd == 5:
		D0 = float(input())
		n = float(input())
		Fshift = fourier(input_img)
		filter = butterWorthLowpass(Fshift, D0, n)
		g = applyFilter(Fshift, filter)

	if cmnd == 6:
		D0 = float(input())
		n = float(input())
		Fshift = fourier(input_img)
		filter = butterWorthHighpass(Fshift, D0, n)
		g = applyFilter(Fshift, filter)

	if cmnd == 7:
		return

	if cmnd == 8:
		return 
	
	print(f'%.4f' % rmse(g, expected_img))



if __name__ == "__main__":
  main()