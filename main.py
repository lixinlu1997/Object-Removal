import sys
import argparse
import mask as M
from yolo import YOLO, detect_video
from PIL import Image
from filling import Object_Remover
from skimage.io import imread, imsave
from patchsize import get_patchsize

while(1):
	print("=====Selece the mode you want to use=====")
	print("---I. input '1' if you don't have a mask and need to detect objects.")
	print("---II. input '2' if you have had a mask.")
	choose = input('---Choose your mode:')

	if choose == str(1):
		yolo = YOLO()
		filename = input('Input the image name:')
		filepath = 'image/' + filename
		ps = get_patchsize(filepath)

		# Get the bonding boxed of objects in the image
		def get_boxes(filename):
			try:
				image = Image.open(filepath)
			except:
				print('Open Error!')
			else:
				r_image,boxes = yolo.detect_image(image)
				r_image.show()
			return boxes

		boxes = get_boxes(filepath)

		box_index = input('Input the indexes you want to remove:')
		box_index = box_index.split(',')

		print(box_index)

		# Create the mask image for the objects selected.
		mask = M.create_mask(filepath)

		for index in box_index:
			index = int(index)
			mask = M.add_mask(mask,boxes[index][0],boxes[index][1],boxes[index][2],boxes[index][3])

		mask_name = 'mask/' + filename.split('.')[0] + '_mask.jpg'
		M.save_mask(mask,mask_name)
	elif choose == str(2):
		filename = input('Input the image name:')
		filepath = 'image/' + filename
		ps = get_patchsize(filepath)
		m_name = input('Input the mask name:')
		mask_name = 'mask/' + m_name
	else:
		print("Enter the correct mode(0 or 1)!")
		assert(0)

	image = imread(filepath)
	mask = imread(mask_name, as_gray=True)
	output_image = Object_Remover(
			image_name =  filename,
      	  image = image,
      	  mask = mask,
     	   patch_size=ps,
     	   show_progress=True,
      	  save_progress=False
   	 ).main()
	dec = input("Do you want to process next image? (y/n)")
	if dec=='y':
		continue
	elif dec=='n':
		break
	else:
		print('Wrong command, exit!')
		break

