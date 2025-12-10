import matplotlib.pyplot as plt
import numpy as np
import cv2

pts = np.load('./point.npy')
pts = pts * 10000. / 255.
color0255 = np.load('./color0255.npy')
color01 = np.load('./color01.npy')

ceiling_idx = np.where(((color0255[:,0] == 8) & (color0255[:,1] == 255) & (color0255[:,2] == 214)))
color0255 = np.delete(color0255, ceiling_idx, axis=0)
color01 = np.delete(color01, ceiling_idx, axis=0)
pts = np.delete(pts, ceiling_idx, axis=0)

floor_idx = np.where(((color0255[:,0] == 255) & (color0255[:,1] == 194) & (color0255[:,2] == 7)))
color0255 = np.delete(color0255, floor_idx, axis=0)
color01 = np.delete(color01, floor_idx, axis=0)
pts = np.delete(pts, floor_idx, axis=0)

rug_idx = np.where(((color0255[:,0] == 255) & (color0255[:,1] == 153) & (color0255[:,2] == 0)))
color0255 = np.delete(color0255, rug_idx, axis=0)
color01 = np.delete(color01, rug_idx, axis=0)
pts = np.delete(pts, rug_idx, axis=0)


high_idx = np.where(((pts[:,1] > -0.5)))
color0255 = np.delete(color0255, high_idx, axis=0)
color01 = np.delete(color01, high_idx, axis=0)
pts = np.delete(pts, high_idx, axis=0)


high_idx = np.where(((pts[:,1] < -1.45)))
color0255 = np.delete(color0255, high_idx, axis=0)
color01 = np.delete(color01, high_idx, axis=0)
pts = np.delete(pts, high_idx, axis=0)


plt.scatter(pts[:,2], pts[:, 0], c = color01, marker='.', s=2)
plt.axis('off')
plt.savefig('map.png')
plt.show()

