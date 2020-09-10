import numpy as np

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  # flip the filter
  filter = filter[-1:-filter.shape[0]-1:-1 , -1:-filter.shape[1]-1:-1] # np.flip(filter) does the same thing

  
  vertical_pad = int((filter.shape[0] - 1) / 2)
  horizontal_pad = int((filter.shape[1] - 1) / 2)
  padded_image = np.pad(image,((vertical_pad,vertical_pad),(horizontal_pad,horizontal_pad),(0,0)), 'reflect')
  
  filtered_image = np.zeros(padded_image.shape, float)
  for dim in range(0 , padded_image.shape[2]):
    for kx in range(vertical_pad , padded_image.shape[0] - vertical_pad):
      for ky in range(horizontal_pad , padded_image.shape[1] - horizontal_pad):
        filtered_image[kx,ky,dim] = np.sum(np.multiply(padded_image[kx - vertical_pad : kx + vertical_pad + 1
          , ky - horizontal_pad : ky + horizontal_pad + 1 , dim], filter))

  filtered_image = filtered_image[vertical_pad:padded_image.shape[0] - vertical_pad , 
  horizontal_pad:padded_image.shape[1] - horizontal_pad , :]

  return filtered_image



def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)

  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  low_frequencies = my_imfilter(image1, filter)
  high_frequencies = image2 - my_imfilter(image2, filter);
  hybrid_image = np.clip(low_frequencies + high_frequencies, 0 , 1)

  return low_frequencies, high_frequencies, hybrid_image
