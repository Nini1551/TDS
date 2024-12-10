# Import des modules
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from scipy import signal
from skimage import color, measure, morphology, filters

def get_image_at_correct_shape(image):
    color_shape_length = image.shape[2]
    assert color_shape_length >= 3, "Image must have at least 3 channels"
    if color_shape_length > 3:
        image = image[:,:,:3]
    return image
    
def get_binaried_image(image, threshold):
    image = get_image_at_correct_shape(image)
    image = color.rgb2gray(image)
    image = image > threshold
    
    return image

def get_image_with_biggest_object(bin_image):
    labels = measure.label(bin_image)
    regions = measure.regionprops(labels)
    idx_max_area = max(enumerate(regions), key=lambda x: x[1].area)[0] + 1
    return np.isin(labels, idx_max_area)

def get_smoothed_image(image, sigma):
    image = morphology.dilation(image, morphology.disk(sigma))
    image = morphology.erosion(image, morphology.disk(sigma))
    return image

def get_binaried_number_image(image, threshold, sigma):
    bin_image = get_binaried_image(image, threshold)
    bin_image = get_image_with_biggest_object(bin_image)
    bin_image = get_smoothed_image(bin_image, sigma)
    return bin_image

def get_number_from_image(number_bin_image):
    correlation_factors = np.zeros(10)
    for i in range(10):
        h = plt.imread(f'./img/ocr/bin{i}.png').copy()
        h = get_binaried_image(h, 0.5).astype(int)

        I = signal.correlate2d(number_bin_image, h, mode='same')

        correlation_factors[i] = np.max(I)
    
    return np.argmax(correlation_factors)

fig, axs = plt.subplots(8, 1, figsize=(4, 20))

for i in range(8):
  I = plt.imread(f'./img/data/{i+1}.png').copy()
  bin_I = get_binaried_image(I, 0.38)
  labels = measure.label(bin_I)
  regions = measure.regionprops(labels)
  filtered_regions = [region for region in regions if region.eccentricity < 0.8]
  sorted_regions = sorted(filtered_regions, key=lambda r: r.area, reverse=True)[:6]
  sorted_regions = sorted(sorted_regions, key=lambda r: r.centroid[1])
  numbers_labels = {region.label for region in sorted_regions}
  filtered_I = np.isin(labels, list(numbers_labels))

  nombre_final = 0
  for region in sorted_regions:
    number_I = np.isin(labels, region.label)
    number = get_number_from_image(number_I)
    print(number)
    nombre_final = nombre_final * 10 + number
    
  print(nombre_final)
    

  axs[i].imshow(filtered_I, cmap='gray')
  axs[i].set_title(f'compteur-{i+1} : {nombre_final}')
  axs[i].axis('off')

plt.show()