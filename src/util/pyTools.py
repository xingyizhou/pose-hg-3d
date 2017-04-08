import sys
import h5py
import numpy as np
import cv2

def getData(tmpFile):
  data = h5py.File(tmpFile, 'r')
  d = {}
  for k, v in data.items():
    d[k] = np.asarray(data[k])
  data.close()
  return d

def ShowImg(data):
  img = data['img']
  img[0], img[2] = img[2].copy(), img[0].copy()
  if img.shape[0] == 3:
    img = img.transpose(1, 2, 0)
  cv2.imshow('img', img)

J = 16
edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
         [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
         [6, 8], [8, 9]]

def show_3d(ax, points, c = (255, 0, 0)):
  points = points.reshape(J, 3)
  points[7] = points[8]
  x, y, z = np.zeros((3, J))
  for j in range(J):
    x[j] = points[j, 0] 
    y[j] = - points[j, 1] 
    z[j] = - points[j, 2] 
  ax.scatter(z, x, y, c = c)
  for e in edges:
    ax.plot(z[e], x[e], y[e], c =c)
    
    
def show_2d(img, points, c = 'g'):
  print 'imgshape', img.shape
  points = points.reshape(J, 2)
  for j in range(J):
    cv2.circle(img, (int(points[j, 0]), int(points[j, 1])), 3, c, -1)
  for e in edges:
    cv2.line(img, (int(points[e[0], 0]), int(points[e[0], 1])), 
                  (int(points[e[1], 0]), int(points[e[1], 1])), c, 2)

def Show3d(data):
  joint = data['joint']
  import matplotlib.pyplot as plt
  import mpl_toolkits.mplot3d
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot((111),projection='3d')
  ax.set_xlabel('z') 
  ax.set_ylabel('x') 
  ax.set_zlabel('y')
  oo = max(joint.max(), oo) / 8
  xmax, ymax, zmax, xmin, ymin, zmin = oo, oo, oo, -oo, -oo, -oo
  if 'gt' in data:
    show_3d(ax, data['gt'], 'r')
  show_3d(ax, joint, 'b')
  max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
  Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
  Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
  Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
  for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([zb], [xb], [yb], 'w')
  if 'img' in data:
    img = data['img'].copy()
    img[0], img[2] = img[2].copy(), img[0].copy()
    if img.shape[0] == 3:
      img = img.transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8).copy()
    show_2d(img, joint.reshape(J, 3)[:, :2], (255, 0, 0))
    if 'gt' in data:
        show_2d(img, data['gt'].reshape(J, 3)[:, :2], (0, 0, 255))
    cv2.imshow('img', img)
  plt.show()

if __name__ == '__main__':
  tmpFile = sys.argv[2]
  data = getData(tmpFile)
  eval(sys.argv[1] + '(data)')
