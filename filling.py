from imageio import imread, imwrite
from imageio import mimsave
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import convolve
from skimage.color import rgb2grey, rgb2lab
from skimage.filters import laplace
import time
import os

'''
For the sake of convenience, we create a class structure to 
store the member function and varuables.
'''
class Object_Remover():
    def __init__(self, image_name, image, mask,\
            patch_size=9,show_progress=True,save_progress=False):
        self.image = image.astype('uint8')  # Convert image to be int.
        self.mask = mask.round().astype('uint8') # Convert mask to be int.
        '''
        The variable patch_size is the size of the template window, 
        default to be 9 and when using, we can set it to be slightly 
        larger than the largest distinguishable texture element.
        '''
        self.patch_size = patch_size
        self.show_progress = show_progress
        self.save_progress = save_progress
        # Intermediate variables
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.temp_image = None
        self.temp_mask = None
        self.history = []
        self.image_name = image_name.split('.')[0]
        '''
        Front is the boundary that is around the missing part of image.
        '''
        self.front = None
        '''
        The confidence of a pixel reflects the confidence in the 
        pixel value and is frozen once the pixel is filled.
        '''
        self.confidence = None
        '''
        The data of a pixel is used to compute the priority of a pixel.
        '''
        self.data = None
        self.priority = None

    def initialize(self):
        '''
        Initialization of the intermediate variables.
        '''
        h,w,_ = self.image.shape
        # The confidence value is 1 at filled pixels and 0 at blank pixels, which 
        # is the inverse of mask
        self.confidence = (1 - self.mask).astype(float)
        self.data = np.zeros((h,w))
        self.temp_image = np.copy(self.image)
        self.temp_mask = np.copy(self.mask)
        self.lab = rgb2lab(self.temp_image)
        self.total_missing = self.temp_mask.sum()


    def plot_image(self,index):
        inverse_mask = 1 - self.temp_mask
        inverse_mask = inverse_mask.reshape(self.height,self.width,1)\
                .repeat(3,axis=2)
        image = self.temp_image*(inverse_mask)

        #image[:, :, 0] += self.front * 255

        # Fill the inside of the target region with white color.
        white_region = (self.temp_mask - self.front) * 255
        white_region = white_region.reshape(self.height,self.width,1)\
                .repeat(3,axis=2)
        image += white_region
        self.history.append(image)
        
        plt.clf()
        plt.imshow(image)
        plt.draw()
        plt.pause(0.0001)

    def get_front(self):
        '''
        Use laplace to find the fron line (the edge of the blank regions).
        '''
        self.front = (laplace(self.temp_mask)>0).astype('uint8')

    def get_patch_index(self,i,j):
        half_size = (self.patch_size-1)//2
        patch_index = [
            [
                max(0,i - half_size),min(i + half_size, self.height-1)
            ],
            [
                max(0,j - half_size),min(j + half_size, self.width-1)
            ]
        ]
        if patch_index[0][1]-patch_index[0][0] != self.patch_size-1:
            if patch_index[0][0] == 0:
                patch_index[0][1] = self.patch_size-1
            if patch_index[0][1] == self.height-1:
                patch_index[0][0] = self.height-self.patch_size
        if patch_index[1][1]-patch_index[1][0] != self.patch_size-1:
            if patch_index[1][0] == 0:
                patch_index[1][1] = self.patch_size-1
            if patch_index[1][1] == self.width-1:
                patch_index[1][0] = self.width-self.patch_size
        return patch_index

    def get_patch_data(self,data,pi):
        return data[pi[0][0]:pi[0][1]+1,pi[1][0]:pi[1][1]+1]

    def update_confidence(self):
        for i, j in np.transpose(np.nonzero(self.front != 0)):
            patch_index = self.get_patch_index(i,j)
            patch_data = self.get_patch_data(self.confidence,patch_index)

            self.confidence[i][j] = np.sum(patch_data)/(self.patch_size**2)
    
    def normal_matrix(self):
        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        x_normal = convolve(self.temp_mask.astype(float), x_kernel)
        y_normal = convolve(self.temp_mask.astype(float), y_kernel)
        normal = np.dstack((x_normal, y_normal))

        height, width = normal.shape[:2]
        norm = np.sqrt(y_normal**2 + x_normal**2)\
                .reshape(height,width,1).repeat(2, axis=2)
        norm[norm == 0] = 1

        unit_normal = normal/norm
        return unit_normal

    def gradient_matrix(self):
        height, width = self.temp_image.shape[:2]

        grey_image = rgb2grey(self.temp_image)
        grey_image[self.temp_mask == 1] = None

        gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
        max_gradient = np.zeros([height, width, 2])

        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self.get_patch_index(point[0],point[1])
            patch_y_gradient = self.get_patch_data(gradient[0], patch)
            patch_x_gradient = self.get_patch_data(gradient[1], patch)
            patch_gradient_val = self.get_patch_data(gradient_val, patch)

            patch_max_pos = np.unravel_index(
                    patch_gradient_val.argmax(),
                    patch_gradient_val.shape

                    )
            max_gradient[point[0], point[1]] = \
                    np.array([patch_y_gradient[patch_max_pos],\
                    patch_x_gradient[patch_max_pos]])
        return max_gradient

    def update_data(self):
        normal = self.normal_matrix()
        gradient = self.gradient_matrix()
        normal_gradient = normal*gradient
        self.data = np.linalg.norm(normal_gradient, axis=2) + 0.001

    def update_priority(self):
        self.update_confidence()
        self.update_data()
        self.priority = self.confidence * self.data * self.front
        #self.priority = self.data * self.front

    def get_next_center_point(self):
        return np.unravel_index(self.priority.argmax(), self.priority.shape)

    def source_patch(self,target_pixel):
        target_patch = self.get_patch_index(target_pixel[0],target_pixel[1])
        # Get potential update region
        front_index = np.transpose(np.nonzero(self.front==1))
        if len(front_index) == 0:
            return None
        bd = (max(0,front_index[0,0]-self.patch_size),\
                min(self.height,front_index[-1,0]+self.patch_size),\
                max(0, min(front_index[:,1])-self.patch_size),\
                min(self.width, max(front_index[:,1])+self.patch_size))

        # Update lab_image
        lab_image = rgb2lab(self.temp_image[bd[0]:bd[1], bd[2]:bd[3]]) 
        self.lab[bd[0]:bd[1], bd[2]:bd[3]] = lab_image
        lab_image = self.lab
        #lab_image = rgb2lab(self.temp_image)

        # Get target mask data
        mask = 1 - self.get_patch_data(self.temp_mask, target_patch)
        rgb_mask = mask.reshape(mask.shape[0],mask.shape[1],1)
        target_data = self.get_patch_data(lab_image,target_patch) * rgb_mask

        # Shape stuff
        temp_shape = (self.height-self.patch_size+1,self.width-self.patch_size+1)
        temp_size = temp_shape[0]*temp_shape[1]
        hs = (self.patch_size-1)//2

        # Calculate Euclidiean distance(spatial proximity)
        source_loc = np.zeros((temp_shape[0], temp_shape[1],2))
        target_loc = np.array([target_patch[0][0], target_patch[1][0]])
        row = np.arange(temp_shape[0]).reshape(-1, 1)
        col = np.arange(temp_shape[1]).reshape(1, -1)
        source_loc[:,:,0] = source_loc[:,:,0] + row
        source_loc[:,:,1] = source_loc[:,:,1] + col
        euclidiean_distance = np.linalg.norm(source_loc-target_loc, axis=2)

        # Calculate square distance (similarity in color)
        conv_square = convolve(lab_image**2,\
                np.flip(rgb_mask**2))[hs:-hs,hs:-hs].sum(axis=2)
        conv_cross = convolve(lab_image,\
                np.flip(target_data*rgb_mask))[hs:-hs,hs:-hs,1]
        squared_distance = (conv_square-2*conv_cross+np.sum(target_data**2))
        squared_distance[bd[0]:bd[1],bd[2]:bd[3]] = np.inf

        # Get final distance and get the most similar patch
        dis = squared_distance + euclidiean_distance
        i, j = np.unravel_index(dis.argmin(), dis.shape)
        return [[i,i+self.patch_size-1],[j,j+self.patch_size-1]]


    def update_image(self,target_pixel,source_patch):
        target_patch = self.get_patch_index(target_pixel[0],target_pixel[1])
        unfilled_pixels = np.argwhere(self.get_patch_data(self.temp_mask,target_patch)==1)+[target_patch[0][0],target_patch[1][0]]
        temp_confidence = self.confidence[target_pixel[0],target_pixel[1]]
        for pixel in unfilled_pixels:
            self.confidence[pixel[0],pixel[1]] = temp_confidence

        mask = self.get_patch_data(self.temp_mask,target_patch)
        mask = mask.reshape(mask.shape[0],mask.shape[1], 1).repeat(3, axis=2)
        source_data = self.get_patch_data(self.temp_image,source_patch)
        target_data = self.get_patch_data(self.temp_image,target_patch)

        new_data = source_data*mask + target_data*(1-mask)

        self.temp_image[target_patch[0][0]:target_patch[0][1]+1,target_patch[1][0]:target_patch[1][1]+1] = new_data
        self.temp_mask[target_patch[0][0]:target_patch[0][1]+1,target_patch[1][0]:target_patch[1][1]+1] = 0

    def if_finished(self):
        remain = self.temp_mask.sum()
        perc = (1-remain/self.total_missing)*100
        print('\r{}% finished.'.format(np.round(perc,2)),end='')
        return remain==0


    def main(self):
        t0 = time.time()
        self.initialize()
        index = 0
        end = False
        while True:
            self.get_front()
            if self.show_progress:
                self.plot_image(index)
            if end:
                break
            self.update_priority()
            target_pixel = self.get_next_center_point()
            source_patch = self.source_patch(target_pixel)
            if type(source_patch) != list:
                break
            self.update_image(target_pixel,source_patch)
            index += 1
            if self.if_finished():
                end = True
        print("Total time: {}".format(time.time()-t0))
        imwrite('output/{}.jpg'.format(self.image_name),self.history[-1])
        if len(self.history) <= 100:
            mimsave('output/{}.gif'.format(self.image_name),self.history)
        else:
            history = [self.history[i] for i in range(0, len(self.history), len(self.history)//100)]
            mimsave('output/{}.gif'.format(self.image_name),history)
        return self.temp_image
