import cv2
import transform_land


tl = transform_land.transform_land()

image = cv2.imread("tes.png")

image, bird_image = transform_land.plot_region(tl, image)

cv2.imwrite("res1.png", image)
cv2.imwrite("res2.png", bird_image)