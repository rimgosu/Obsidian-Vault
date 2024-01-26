import numpy as np, cv2
from  Common.filters import filter

def differential(image, data1, data2):
    

    
    return dst, dst1, dst2

image = cv2.imread("Source/chap07/images/edge.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")
    
data1 = [-1, 0, 0,
          0, 1, 0,
          0, 0, 0]
data2 = [ 0, 0, -1,
          0, 1, 0,
          0, 0, 0]
dst, dst1, dst2 = differential(image, data1, data2)  		# 회선 수행 및 두 방향의 크기 계산

cv2.imshow("image", image)
cv2.imshow("roberts edge", dst)
cv2.imshow("dst1", dst1)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)