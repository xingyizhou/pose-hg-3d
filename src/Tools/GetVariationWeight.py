import numpy as np
'''
5. Standard bone length:
E_Bone_Knee2Ankle			339.876159825706f
E_Bone_Hip2Knee				409.4893491875976f
E_Bone_Pelvis2Hip			132.693604f
E_Bone_Pelvis2Thorax		438.931000f
E_Bone_Neck2Thorax			52.8063011f
E_Bone_Head2Neck			185.185272f
E_Bone_Elbow2Wrist			246.979111f
E_Bone_Shoulder2Elbow		280.904663f
E_Bone_Thorax2Shoulder		145.347519f
'''
shape = [
339.876159825706,
409.4893491875976,
132.693604,
438.931000,
52.8063011,
185.185272,
246.979111,
280.904663,
145.347519,
438.931000 + 52.8063011
]

'''
shape = [452.125519758,
456.008604234,
134.201602044,
421.517337247,
77.938555657,
182.550526846,
249.772589249,
284.125346475,
180.776851782, 
421.517337247 + 77.938555657
]

  typedef enum {
    Knee2Ankle = 0, Hip2Knee, Pelvis2Hip, 
    Pelvis2Torso, Torso2Neck, Neck2Head, 
    Elbow2Wrist, Shoulder2Elbow, Torso2Shoulder, 
    NUM_SHAPE_PARAMETERS
  } Shape;

    self.skeletonRef = {{1,2,1},    {2,3,1},    {3,7,1},
                        {4,5,2},    {4,7,2},    {5,6,2},
                        {7,9,0},    {9,10,0},
                        {13,9,3},   {11,12,3},  {12,13,3},
                        {14,9,4},   {14,15,4},  {15,16,4}}
'''

Knee2Ankle = 0
Hip2Knee = 1
Pelvis2Hip = 2
Pelvis2Torso = 3
Torso2Neck = 4
Neck2Head = 5
Elbow2Wrist = 6
Shoulder2Elbow = 7
Torso2Shoulder = 8

skeletonShapeId = [Knee2Ankle, Hip2Knee, Pelvis2Hip, 
                   Hip2Knee, Pelvis2Hip, Knee2Ankle, 
                   9, Neck2Head, 
                   Torso2Shoulder, Elbow2Wrist, Shoulder2Elbow, 
                   Torso2Shoulder, Shoulder2Elbow, Elbow2Wrist]
                   
N = len(skeletonShapeId)
mean = 0
for i in range(N):
  mean += shape[skeletonShapeId[i]] / N

weight = np.zeros(N)  
for i in range(N):
  weight[i] = mean / shape[skeletonShapeId[i]]
  print '{},'.format(weight[i]), 

