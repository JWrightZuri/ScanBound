import cv2
import numpy as np

img = cv2.imread(r"C:\Users\wrja\Desktop\STM_Data\Orito_Reaction\PdGaA(111)_Pd1_CX3-2\20241107_PdGaA111_279a_8min_-175C_RT\saved_output\region20007_Z.jpg")

# Create sharpening kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
# kernel = kernel * 0.5
sharpened = cv2.filter2D(img, -1, kernel)
# cv2.imwrite("sharpened.jpg", sharpened)
cv2.imshow("Original", img)
cv2.imshow("Sharpened", sharpened)
cv2.waitKey(0)
# cv2.destroyAllWindows()