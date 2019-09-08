from imageio import imread, imwrite
import numpy as np


def create_mask(filename):
	img = imread(filename)
	mask = np.zeros(img.shape)
	return mask

def add_mask(mask,top,bot,left,right):
	for i in range(3):
		for j in range(max(0,top-3),min(mask.shape[0]-1,bot+4)):
			for k in range(max(0,left-3),min(mask.shape[1]-1,right+4)):
				mask[j][k][i] = 255
	return mask

def save_mask(mask,mask_name):
	imwrite(mask_name,mask)
