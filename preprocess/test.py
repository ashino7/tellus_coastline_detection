import cv2
import json
import numpy as np
import tifffile
from util.resize_view import resize_view

import pickle
import matplotlib.pyplot as plt

idx = 0
data = tifffile.imread(f'input/train_images/train_{idx:02d}.tif')
with open(f"input/train_annotations/train_{idx:02d}.json", "r") as fp:
    annotation = json.load(fp)

# plt.title(f"train {idx:02d}")
# plt.imshow(data)
# plt.colorbar()
# plt.show()

plt.title(f"train {idx:02d}")
plt.imshow(np.log10(data + 1.0e-1), cmap="ocean")
for line in annotation['validate_lines']:
    ix = [line[i][0] for i in range(2)]
    iy = [line[i][1] for i in range(2)]
    plt.plot(ix, iy)
for point in annotation['coastline_points']:
    plt.plot(point[0], point[1], ",", color="red")
plt.show()

c=0