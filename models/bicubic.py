import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import rasterio

import math
import sys

def u(s, a):
    s = abs(s)

    if 0 <= s <= 1:
        return (a + 2) * s**3 - (a + 3) * s**2 + 1
    elif 1 < s <= 2:
        return a * s**3 - 5 * a * s**2 + 8 * a * s - 4 * a
    else:
        return 0.0

# Padding
def padding(img, C, H, W):
    zimg = np.zeros((C, H+4, W+4,), dtype=img.dtype)
    zimg[:C, 2:H+2, 2:W+2] = img # ZERO PADDING
    
    # Pad the first/last two col and row
    zimg[:, 2:H+2, 0:2] = img[:C, :, 0:1] # first column + org img px
    zimg[:, H+2:H+4, 2:W+2] = img[:, H-1:H, :] #
    zimg[:, 2:H+2, W+2:W+4] = img[:, :, W-1:W]
    zimg[:, 0:2, 2:W+2] = img[:C, 0:1, :]
    
    # Pad the missing eight points
    zimg[:, 0:2, 0:2] = img[:, 0:1, 0:1]
    zimg[:, H+2:H+4, 0:2] = img[:, H-1:H, 0:1]
    zimg[:, H+2:H+4, W+2:W+4] = img[:, H-1:H, W-1:W]
    zimg[:, 0:2, W+2:W+4] = img[:, 0:1, W-1:W]
    
    return zimg

# Bicubic operation
def bicubic(img, ratio, a):

    # Get image size
    C, H, W = img.shape

    # Here H = Height, W = weight,
    # C = Number of channels if the
    # image is coloured.
    img = padding(img, C, H, W)

    # Create new image
    dH = math.floor(H*ratio)
    dW = math.floor(W*ratio)

    # Converting into matrix
    dst = np.zeros((C, dH, dW))

    # np.zeroes generates a matrix
    # consisting only of zeroes
    # Here we initialize our answer
    # (dst) as zero

    h = 1/ratio

    print('Starting bicubic interpolation')
    inc = 0

    for c in range(C):
        for j in range(dH):
            for i in range(dW):

                # Getting the coordinates of the
                # nearby values
                x, y = i * h + 2, j * h + 2

                fx = math.floor(x)
                fy = math.floor(y)

                x1 = 1 + x - fx
                x2 = x - fx
                x3 = fx + 1 - x
                x4 = fx + 2 - x

                y1 = 1 + y - fy
                y2 = y - fy
                y3 = fy + 1 - y
                y4 = fy + 2 - y

                # Considering all nearby 16 values
                mat_l = np.array([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]])
                mat_m = np.array([[img[c, int(y-y1), int(x-x1)],
                                    img[c, int(y-y2), int(x-x1)],
                                    img[c, int(y+y3), int(x-x1)],
                                    img[c, int(y+y4), int(x-x1)]],
                                   [img[c, int(y-y1), int(x-x2)],
                                    img[c, int(y-y2), int(x-x2)],
                                    img[c, int(y+y3), int(x-x2)],
                                    img[c, int(y+y4), int(x-x2)]],
                                   [img[c, int(y-y1), int(x+x3)],
                                    img[c, int(y-y2), int(x+x3)],
                                    img[c, int(y+y3), int(x+x3)],
                                    img[c, int(y+y4), int(x+x3)]],
                                   [img[c, int(y-y1), int(x+x4)],
                                    img[c, int(y-y2), int(x+x4)],
                                    img[c, int(y+y3), int(x+x4)],
                                    img[c, int(y+y4), int(x+x4)]]])
                mat_r = np.array([[u(y1, a)], 
                                  [u(y2, a)], 
                                  [u(y3, a)], 
                                  [u(y4, a)]])
                
                # Here the dot function is used to get the dot 
                # product of 2 matrices
                # dst[c, j, i] = np.dot(np.dot(mat_l, mat_m), mat_r)
                dst[c, j, i] = (mat_l @ mat_m @ mat_r).item()

    # If there is an error message, it
    # directly goes to stderr
    sys.stderr.write('\n')
    
    # Flushing the buffer
    sys.stderr.flush()
    return dst

def plot_image(org_img, bicubic_img):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(org_img.transpose(1, 2, 0))
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    # If necessary, clip values to the valid range
    bicubic_img = np.clip(bicubic_img, 0, 255).astype(np.uint8)
    plt.imshow(bicubic_img.transpose(1, 2, 0))
    plt.title('Bicubic Image')
    plt.show()

def main():
    # Read the image
    ip_path="H:\\Dual-SR-Dataset-Patched\\patch_dataset\\train\\LR_1m\\img_001\\patch_001.tif"
    with rasterio.open(ip_path) as src:
        img=src.read()

    # Bicubic interpolation
    ratio = 4
    a = -0.5
    dst = bicubic(img, ratio, a)

    # Plot the original and bicubic images
    plot_image(img, dst)

    # Save the image
    # op_path="/kaggle/working/bicubic_output.tif"
    # with rasterio.open(op_path, 'w', driver='GTiff', height=dst.shape[1], width=dst.shape[2], count=dst.shape[0], dtype=dst.dtype) as dst_file:
    #     dst_file.write(dst)

if __name__ == "__main__":
    main()