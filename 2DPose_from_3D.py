import json
import numpy as np
import cv2 as cv
import math
import sympy as sp



class humanPoint:
	def __init__(self, Points):
		self.points = Points

class photoHuman:
	def __init__(self):
		self.human = []


#ground truth
gdTruthFile = open("./data/camma_mvor_2018.json", "rb")
gdTruthJson = json.load(gdTruthFile)

cam3d = gdTruthJson["annotations3D"]

#cam_info
R2 = np.array([[-0.98275320616549,
        	0.044883688744506,
         	0.1793922803694],
          	[0.184830538935,
          	0.26891529014386,
          	0.94526305259639],
          	[-0.0058144344906895,
          	0.96211746747033,
          	-0.27257323995584]])
R3 = np.array([[0.32853737750458,
          	-0.53718216533788,
          	0.77685166719607],
          	[0.54278875050447,
          	0.78050206600633,
          	0.3101562465688],
          	[-0.7729447353519,
          	0.31976842590029,
          	0.54800053821966]])
t2 = np.array([-496.23761858706,
          	-2260.7556455482,
          	3156.0043173695])
t3 = np.array([-2495.6127218428,
          	-639.27778089763,
          	908.30818963998])
K1 = np.matrix([[538.597, 0, 315.8367],
			[0, 538.2393, 241.9166],
			[0, 0, 1]])
K2 = np.matrix([[534.8386, 0, 321.2326],
			[0, 534.4008, 243.3514],
			[0, 0, 1]])
K3 = np.matrix([[541.4062, 0, 323.9545],
			[0, 540.5641, 238.6629],
			[0, 0, 1]])

T1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
T2 = np.c_[R2,np.transpose([t2])]
T3 = np.c_[R3,np.transpose([t3])]

#calculate the 2d ground truth
def cam2d(xx, yy, zz):
	if int(xx) == 0 & int(yy) == 0 & int(zz) == 0:
		return np.array([[0]]), np.array([[0]]), np.array([[0]]), np.array([[0]]), np.array([[0]]), np.array([[0]])
	point = np.array([xx, yy, zz, 1])
	#point = np.dot(camToRef, np.transpose([point]))
	#cam1
	cc1 = np.dot(T1, np.transpose([point]))
	pp1 = np.dot(K1, cc1) / cc1[2]

	#cam2
	cc2 = np.dot(T2, np.transpose([point]))
	pp2 = np.dot(K2, cc2) / cc2[2]

	#cam3
	cc3 = np.dot(T3, np.transpose([point]))
	pp3 = np.dot(K3, cc3) / cc3[2]

	return np.array(pp1[0]), np.array(pp1[1]), np.array(pp2[0]), np.array(pp2[1]), np.array(pp3[0]), np.array(pp3[1])

day1cam1photo = [photoHuman() for i in range(57)]
day1cam2photo = [photoHuman() for i in range(57)]
day1cam3photo = [photoHuman() for i in range(57)]

day2cam1photo = [photoHuman() for i in range(330)]
day2cam2photo = [photoHuman() for i in range(330)]
day2cam3photo = [photoHuman() for i in range(330)]

day3cam1photo = [photoHuman() for i in range(223)]
day3cam2photo = [photoHuman() for i in range(223)]
day3cam3photo = [photoHuman() for i in range(223)]

day4cam1photo = [photoHuman() for i in range(122)]
day4cam2photo = [photoHuman() for i in range(122)]
day4cam3photo = [photoHuman() for i in range(122)]

#read cam1 ground truth
for cam in cam3d:
	photo = int(cam["image_ids"][8:11])
	keyList = cam["keypoints3D"]
	if keyList == [0] * 48:
		continue
	# p = []
	cam12D = []
	cam22D = []
	cam32D = []
	for i in range(10):
		# p.append(keyList[4 * i])
		# p.append(keyList[4 * i + 1])
		# p.append(keyList[4 * i + 2])
		point2d = cam2d(keyList[4 * i], keyList[4 * i + 1], keyList[4 * i + 2])
		#print(len(point2d[0].tolist()))
		cam12D.append(point2d[0].tolist()[0][0])
		cam12D.append(point2d[1].tolist()[0][0])
		cam22D.append(point2d[2].tolist()[0][0])
		cam22D.append(point2d[3].tolist()[0][0])
		cam32D.append(point2d[4].tolist()[0][0])
		cam32D.append(point2d[5].tolist()[0][0])
	# test_y.append(np.array(p))
	human1 = humanPoint(cam12D)
	human2 = humanPoint(cam22D)
	human3 = humanPoint(cam32D)
	if int(cam["image_ids"][0]) == 1:
		day1cam1photo[photo].human.append(human1)
		day1cam2photo[photo].human.append(human2)
		day1cam3photo[photo].human.append(human3)

	elif int(cam["image_ids"][0]) == 2:
		day2cam1photo[photo].human.append(human1)
		day2cam2photo[photo].human.append(human2)
		day2cam3photo[photo].human.append(human3)

	elif int(cam["image_ids"][0]) == 3:
		day3cam1photo[photo].human.append(human1)
		day3cam2photo[photo].human.append(human2)
		day3cam3photo[photo].human.append(human3)

	elif int(cam["image_ids"][0]) == 4:
		day4cam1photo[photo].human.append(human1)
		day4cam2photo[photo].human.append(human2)
		day4cam3photo[photo].human.append(human3)

	else:
		raise RuntimeError('testError')

humans = []

for i in range(len(day1cam1photo)):
	human = {}
	human["day"] = 1
	human["photo"] = "%03d" % i
	keypoint1 = []
	keypoint2 = []
	keypoint3 = []
	for person in day1cam1photo[i].human:
		keypoint1.append(person.points)
	for person in day1cam2photo[i].human:
		keypoint2.append(person.points)
	for person in day1cam3photo[i].human:
		keypoint3.append(person.points)
	human["cam1"] = keypoint1
	human["cam2"] = keypoint2
	human["cam3"] = keypoint3
	humans.append(human)

for i in range(len(day2cam1photo)):
	human = {}
	human["day"] = 2
	human["photo"] = "%03d" % i
	keypoint1 = []
	keypoint2 = []
	keypoint3 = []
	for person in day2cam1photo[i].human:
		keypoint1.append(person.points)
	for person in day2cam2photo[i].human:
		keypoint2.append(person.points)
	for person in day2cam3photo[i].human:
		keypoint3.append(person.points)
	human["cam1"] = keypoint1
	human["cam2"] = keypoint2
	human["cam3"] = keypoint3
	humans.append(human)


for i in range(len(day3cam1photo)):
	human = {}
	human["day"] = 3
	human["photo"] = "%03d" % i
	keypoint1 = []
	keypoint2 = []
	keypoint3 = []
	for person in day3cam1photo[i].human:
		keypoint1.append(person.points)
	for person in day3cam2photo[i].human:
		keypoint2.append(person.points)
	for person in day3cam3photo[i].human:
		keypoint3.append(person.points)
	human["cam1"] = keypoint1
	human["cam2"] = keypoint2
	human["cam3"] = keypoint3
	humans.append(human)


for i in range(len(day4cam1photo)):
	human = {}
	human["day"] = 4
	human["photo"] = "%03d" % i
	keypoint1 = []
	keypoint2 = []
	keypoint3 = []
	for person in day4cam1photo[i].human:
		keypoint1.append(person.points)
	for person in day4cam2photo[i].human:
		keypoint2.append(person.points)
	for person in day4cam3photo[i].human:
		keypoint3.append(person.points)
	human["cam1"] = keypoint1
	human["cam2"] = keypoint2
	human["cam3"] = keypoint3
	humans.append(human)


with open('./data/3DPoseTo2D.json', 'w') as f:
    json.dump(humans, f)