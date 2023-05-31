import numpy as np

img = np.array([[1,1,1,0,0],
                [0,1,1,1,0],
                [0,0,1,1,1],
                [0,0,1,1,0],
                [0,1,1,0,0]])

weight = np.array([[1,0,1],
                   [0,1,0],
                   [1,0,1]])

n = img.shape[0] - weight.shape[0]
k = img.shape[1] - weight.shape[1]


# stride = 1
ret = np.zeros((3,3))
img_row_start = 0
img_row_end = weight.shape[0]

for i in range(n+1):
  img_col_start = 0
  img_col_end = weight.shape[1]
  
  # moving across the img
  for j in range(k+1):
    ret[i,j] = np.sum(img[img_row_start:img_row_end, img_col_start:img_col_end] * weight)
    img_col_start += 1
    img_col_end += 1
    
  # moving down the img
  img_row_start += 1
  img_row_end += 1
  
print(ret)
  
  
  
