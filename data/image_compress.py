
import os
import cv2
import numpy as np

local_path = os.path.abspath(".")
_sets = ["train1", "validation1", "validation2", "test1", "test2"]
for _set in _sets :
  path_in = local_path + "/" + _set + "/png/"
  path_out = local_path + "/" + _set + "/png_cp256/"
  N = 100000 if "train" in _set else 20000
  size = (256,256)
  images = []

  for i in range(N) :
    if i % 100 == 0 :
      print(i)
    image = cv2.imread(path_in + str(i) + ".png")
    image = cv2.resize(image, size)
    cv2.imwrite(path_out + str(i) + ".png", image, [int(cv2.IMWRITE_PNG_COMPRESSION ), 3])
