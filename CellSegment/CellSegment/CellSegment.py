import numpy as np
from numpy import linalg
import cv2
import matplotlib.pyplot as plt
import os
from skimage.io import imread
from skimage.exposure import rescale_intensity
from skimage.color import rgb2grey, separate_stains
from PIL import Image

arr=np.arange(1,17,1)


rgb_from_hrd = np.array([[0.644, 0.710, 0.285],
                         [0.0326, 0.873, 0.487],
                         [0.268, 0.570, 0.776]])
#conv_matrix
hrd_from_rgb = linalg.inv(rgb_from_hrd)

for i in range(0,16):

    print(str(arr[i])+'.jpg')
    ihc_rgb = imread('img/'+str(arr[i])+'.jpg')




    # 正規化 0 to 1
    # ihc_hrd[:, :, (0,1, or 2 -- is the color channel)]
    def stainspace_to_2d_array(ihc_xyz, channel):
        rescale = rescale_intensity(ihc_xyz[:, :, channel], out_range=(0,1))
        stain_array = np.dstack((np.zeros_like(rescale), rescale, rescale))
        grey_array = rgb2grey(stain_array)
        return grey_array


    ihc_hrd = separate_stains(ihc_rgb, hrd_from_rgb)

    Hema_Gray_Array = stainspace_to_2d_array(ihc_hrd, 0)
    permred_Gray_Array = stainspace_to_2d_array(ihc_hrd, 1)
    DAB_Grey_Array = stainspace_to_2d_array(ihc_hrd, 2)

    plt.imshow(DAB_Grey_Array, cmap=plt.cm.gray, interpolation='nearest')

    DAB_Grey_Array=DAB_Grey_Array*255

    cv2.imwrite('img/'+str(arr[i])+'_output.jpg',DAB_Grey_Array)

    #Plot images
    fig, axes = plt.subplots(1, 2, figsize=(12, 11))

    ax0, ax1 = axes.ravel()

    ax0.imshow(ihc_rgb, interpolation='nearest')
    ax0.set_title("Original")

    ax1.imshow(DAB_Grey_Array, cmap=plt.cm.gray, interpolation='nearest')
    ax1.set_title("DECONVOLTION")

    #ax2.imshow(permred_Gray_Array, cmap=plt.cm.gray)
    #ax2.set_title("Permanant Red")

    #ax3.imshow(Hema_Gray_Array, cmap=plt.cm.gray)
    #ax3.set_title("Hematoxylin")

    for ax in axes.ravel():
        ax.axis('on')

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    plt.show()




    #Color deconvolution
    #Normalized optical density matrix
    # R      G      B
    # X      X      X   Hematoxylin(0)
    # X      X      X   Red(1)
    # X      X      X   DAB(2)
    #Hematoxylin(0), Red(1), DAB(2)