import numpy as np
import eta.core.image as etai
from scipy.ndimage.filters import convolve
import sys
from skimage.color import rgb2grey, rgb2lab
import cv2


def _calculate_potts_energy(data):
    x_derivative = convolve(data, np.array([[-1,1]]))
    y_derivative = convolve(data, np.array([[-1],[1]]))
    potts_energy = 0
    potts_energy += np.sum(x_derivative != 0)
    potts_energy += np.sum(y_derivative != 0)
    return potts_energy


def _create_gaussian_kernel(sigma, filtersize=20):
    '''Creates a Gaussian kernel.

    Returns:
        func_2d: a 2-D Gaussian kernel
    '''
    x_space = np.linspace(-1, 1, num=filtersize)
    y_space = np.linspace(-1, 1, num=filtersize)
    x_func = np.exp(-(x_space ** 2) / (2 * sigma ** 2))
    y_func = np.exp(-(y_space ** 2) / (2 * sigma ** 2))
    func_2d = y_func[:, np.newaxis] * x_func[np.newaxis, :]
    func_2d = (1 / func_2d.sum()) * func_2d
    return func_2d

def get_patchsize(image):
    print("---Finding the best patch size---")
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    etai.write(img, 'test/a.png')
    out_dir = 'test/'
    diff = [i/10 for i in range(50,100,2)]
    dom = 200
    sigmas = []
    for d in diff:
        if dom <= 0:
            break
        sigmas.append(5/dom)
        dom -= d
    filtered = []
    for sigma in sigmas:
        filt = _create_gaussian_kernel(sigma)
        filtered.append(convolve(img,filt).astype('int'))
    pott_diff = []
    pott = []
    last_pott=0
    for i in range(len(filtered)-1):
        diff = filtered[i+1]-filtered[i]
        pe = _calculate_potts_energy(diff)
        pott_diff.append(pe-last_pott)
        last_pott = pe
        pott.append(pe)
        
        #print("Pott energy of {}.jpg(sigma={}) is: {}"\
        #    .format(i,np.round(sigmas[i],3),np.round(pe,2)))
        
        etai.write(diff*255/max(1, np.max(diff)),out_dir+'{}.jpg'.format(i))
    best_index = np.argmax(np.array(pott_diff)[1:])+1
    #print("Best sigma is: {} in test/{}.jpg.".format(np.round(sigmas[best_index],3), best_index))
    patchsize = (2*np.prod(img.shape)-np.sum(img.shape))//pott[best_index]*2+3
    patchsize = min(patchsize,19)
    print("Best patchsize is {}.".format(patchsize))
    return patchsize

if __name__ == '__main__':
    get_patchsize(sys.argv[1])
