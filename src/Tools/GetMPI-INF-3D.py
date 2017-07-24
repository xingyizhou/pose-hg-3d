import h5py 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import os
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
DEBUG = False

DATA_PATH = '/home/zxy/Datasets/MPI-INF-3DHP/'
IMG_PATH = DATA_PATH + 'images/'
BBOX_PATH = DATA_PATH + 'MPI-INF-3DHP_crop_info/'
ANNOT2D_PATH = DATA_PATH + 'MPI-INF-3DHP_2d_annot/'
SAVE_PATH = '../../data/mpi-inf-3dhp/'
if not os.path.exists(SAVE_PATH):
  os.mkdir(SAVE_PATH)
IMG_SIZE = 368
eps = 1e-4
J = 16
MPII_order = [10, 9, 8, 11, 12, 13, 14, 15, 1, 0, 4, 3, 2, 5, 6, 7]
edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
         [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
         [6, 8], [8, 9]]
         
def show_3d(points, c = 'b'):
  points = points.reshape(J, 3)
  x, y, z = np.zeros((3, J))
  for j in range(J):
    x[j] = points[j, 0] 
    y[j] = - points[j, 1] 
    z[j] = - points[j, 2] 
  ax.scatter(z, x, y, c = c)
  #ax.scatter(z[k], x[k], y[k], c = (0, 0, 0))
  for e in edges:
    ax.plot(z[e], x[e], y[e], c = c)

def show_2d(img, points, c = (255, 0, 0)):
  points = points.reshape(J, 2)
  for j in range(J):
    cv2.circle(img, (int(points[j, 0]), int(points[j, 1])), 3, c, -1)
  for e in edges:
    cv2.line(img, (int(points[e[0], 0]), int(points[e[0], 1])), 
                  (int(points[e[1], 0]), int(points[e[1], 1])), c, 2)



id_ = []
TSId_ = []
data_id_ = []
univ_annot3_ = []
valid_frame_ = []
activity_annotation_ = []
bbox_ = []
annot2d_ = []
num = 0

for TSId in range(1, 7):
  dataFile = DATA_PATH + 'TS{}/bb_cropped_data.mat'.format(TSId)
  bboxFile = BBOX_PATH + 'TS{}_crop_info.mat'.format(TSId)
  annot2DFile = ANNOT2D_PATH + 'TS{}_gt_2d_annot.mat'.format(TSId)
  data = h5py.File(dataFile, 'r')
  bboxData = scipy.io.loadmat(bboxFile)
  annot2dData = scipy.io.loadmat(annot2DFile)
  print(type(data))
  print(data)
  print(data.keys())
  refs = data['#refs#']
  print refs
  activity_annotation = data['activity_annotation']
  images = data['images']
  univ_annot3 = data['univ_annot3']
  valid_frame = data['valid_frame']
  annot_2d = annot2dData['annot2d']
  print(activity_annotation.shape)
  print(univ_annot3.shape)
  print(valid_frame.shape)
  print activity_annotation[0]

  n = images.shape[1]
  for i in range(n):
    '''
    img = (data[(images[0][i])][1] + 0.4)
    img = img.transpose(1, 2, 0).copy()
    for h in range(img.shape[0]):
      for w in range(img.shape[1]):
        if img[h][w][0] < 0.4 + eps and img[h][w][1] < 0.4 + eps and img[h][w][2] < 0.4 + eps and img[h][w][0] > 0.4 - eps and img[h][w][1] > 0.4 - eps and img[h][w][2] > 0.4 - eps:
          img[h][w][0], img[h][w][1], img[h][w][2] = 0, 0, 0
    joint = univ_annot3[i][0][MPII_order]
    img = img * 255
    img_name = '{}_{}'.format(TSId, i + 1)
    cv2.imwrite(IMG_PATH + img_name + '.png', img)
    '''
    
    if num % 100 == 0:
      print num
    if valid_frame[i][0] > 0.5:
      num = num + 1
      id_.append(num)
      TSId_.append(TSId)
      data_id_.append(i + 1)
      univ_annot3_.append(univ_annot3[i][0])
      valid_frame_ .append(valid_frame[i][0])
      activity_annotation_.append(activity_annotation[i][0])
      bbox_.append(bboxData['crop_rectangle'][i][0][0])
      annot2d_.append(annot_2d[i][0])
      #print(bboxData['crop_rectangle'][i][0][0], bboxData['crop_rectangle'][i][0][0].shape)
    
    if DEBUG:
      fig = plt.figure()
      ax = fig.add_subplot((111),projection='3d')
      ax.set_xlabel('z') 
      ax.set_ylabel('x') 
      ax.set_zlabel('y')
      oo = 2
      xmax, ymax, zmax, xmin, ymin, zmin = oo, oo, oo, -oo, -oo, -oo
      joint = (joint - joint[6]) / 5 + IMG_SIZE / 2
      show_3d(joint)
      show_2d(img, joint[:, :2])
      cv2.imshow('img', img)
      cv2.waitKey()
    #plt.show()
print 'num = ', num

h5name = SAVE_PATH + 'annotTest.h5'
f = h5py.File(h5name, 'w')
f['id'] = id_
f['TSId'] = TSId_
f['data_id'] = data_id_
f['univ_annot3'] = univ_annot3_
f['valid_frame'] = valid_frame_
f['activity_annotation'] = activity_annotation_
f['bbox'] = bbox_
f['annot_2d'] = annot2d_
f.close()
