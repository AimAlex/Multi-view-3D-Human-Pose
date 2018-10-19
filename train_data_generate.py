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

class keyPoint3D:
	def __init__(self,xx, yy, zz):
		self.x = xx
		self.y = yy
		self.z = zz

class humanPoint3D:
	def __init__(self, Points):
		self.points = Points

class photoHuman3D:
	def __init__(self):
		self.human = []


#ground truth
gdTruthFile = open("./data/camma_mvor_2018.json", "rb")
gdTruthJson = json.load(gdTruthFile)

cam2d = gdTruthJson["annotations"]

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
for cam in cam2d:
	photo = int(cam["image_id"]) % 1000
	keyList = cam["keypoints"]
	human = humanPoint(keyList)
	if keyList == [0] * 30:
		continue
	if int(cam["image_id"]/10**10) == 1:
		if int(cam["image_id"]/10**7) % 10 == 1:
			day1cam1photo[photo].human.append(human)
		elif int(cam["image_id"]/10**7) % 10 == 2:
			day1cam2photo[photo].human.append(human)
		elif int(cam["image_id"]/10**7) % 10 == 3:
			day1cam3photo[photo].human.append(human)
		else:
			raise RuntimeError('testError')

	elif int(cam["image_id"]/10**10) == 2:
		if int(cam["image_id"]/10**7) % 10 == 1:
			day2cam1photo[photo].human.append(human)
		elif int(cam["image_id"]/10**7) % 10 == 2:
			day2cam2photo[photo].human.append(human)
		elif int(cam["image_id"]/10**7) % 10 == 3:
			day2cam3photo[photo].human.append(human)
		else:
			raise RuntimeError('testError')

	elif int(cam["image_id"]/10**10) == 3:
		if int(cam["image_id"]/10**7) % 10 == 1:
			day3cam1photo[photo].human.append(human)
		elif int(cam["image_id"]/10**7) % 10 == 2:
			day3cam2photo[photo].human.append(human)
		elif int(cam["image_id"]/10**7) % 10 == 3:
			day3cam3photo[photo].human.append(human)
		else:
			raise RuntimeError('testError')

	elif int(cam["image_id"]/10**10) == 4:
		if int(cam["image_id"]/10**7) % 10 == 1:
			day4cam1photo[photo].human.append(human)
		elif int(cam["image_id"]/10**7) % 10 == 2:
			day4cam2photo[photo].human.append(human)
		elif int(cam["image_id"]/10**7) % 10 == 3:
			day4cam3photo[photo].human.append(human)
		else:
			raise RuntimeError('testError')

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


with open('./data/2dPose.json', 'w') as f:
    json.dump(humans, f)
# #read cam2 & cam3 ground truth and match
# for cam in cam2d:
# 	if int(cam["image_id"]/10**10) != 1:
# 		#print(int(cam["image_id"] / 10**10))
# 		continue
# 	view = int(cam["image_id"]/10**7) % 10
# 	if view == 1:
# 		continue

# 	photo = int(cam["image_id"]) % 1000
# 	humanID = cam["person_id"]
# 	if humanID < 0:
# 		continue
# 	pointList = [1] * 10
# 	keyList = cam["keypoints"]

# 	for i in range(10):
# 		if keyList[3 * i + 2] == 0:
# 			pointList[i] = keyPoint(0, 0, 0)
# 			continue
# 		wx = random.gauss(0,subdeviation)
# 		wy = random.gauss(0,subdeviation)
# 		pointList[i] = keyPoint(keyList[3 * i] + wx, keyList[3 * i + 1] + wy, GaussianConf(wx, wy))

# 	#print(pointList)
# 	human = humanPoint(pointList)
# 	#print (gt_cam1photo[1].human[0].cam2, gt_cam1photo[1].human[0].cam3)
# 	#print(humanID, photo, len(gt_cam1photo[photo].human), cam["image_id"])
# 	if view == 2:
# 		if len(gt_cam1photo[photo].human) <= humanID:
# 			faker = humanPoint([keyPoint(0, 0, 0) for i in range(10)])
# 			faker.init_cam1()
# 			gt_cam1photo[photo].human.insert(humanID, faker)
# 		gt_cam1photo[photo].human[humanID].cam2 = human
# 	else :
# 		#print (gt_cam1photo[1].human[0].cam2, gt_cam1photo[1].human[0].cam3, photo, humanID)
# 		if len(gt_cam1photo[photo].human) <= humanID:
# 			faker = humanPoint([keyPoint(0, 0, 0) for i in range(10)])
# 			faker.init_cam1()
# 			gt_cam1photo[photo].human.insert(humanID, faker)
# 		gt_cam1photo[photo].human[humanID].cam3 = human


# cam3d = gdTruthJson["annotations3D"]

# gt_3dphoto = [photoHuman3D() for i in range(57)]

# for cam in cam3d:
# 	str = cam["image_ids"]
# 	if str[0] != '1':
# 		continue
# 	#print(int(str[8:11]))
# 	photo = int(str[8:11])
# 	humanID = cam["person_id"]
# 	pointList = [1] * 10
# 	keyList = cam["keypoints3D"]

# 	pointList[0] = keyPoint3D(keyList[0], keyList[1], keyList[2])
# 	pointList[1] = keyPoint3D(keyList[4], keyList[5], keyList[6])
# 	pointList[2] = keyPoint3D(keyList[8], keyList[9], keyList[10])
# 	pointList[3] = keyPoint3D(keyList[12], keyList[13], keyList[14])
# 	pointList[4] = keyPoint3D(keyList[16], keyList[17], keyList[18])
# 	pointList[5] = keyPoint3D(keyList[20], keyList[21], keyList[22])
# 	pointList[6] = keyPoint3D(keyList[24], keyList[25], keyList[26])
# 	pointList[7] = keyPoint3D(keyList[28], keyList[29], keyList[30])
# 	pointList[8] = keyPoint3D(keyList[32], keyList[33], keyList[34])
# 	pointList[9] = keyPoint3D(keyList[36], keyList[37], keyList[38])

# 	human = humanPoint3D(pointList)
# 	if humanID >= len(gt_3dphoto[photo].human):
# 		gt_3dphoto[photo].human.append(human)
# 	else:
# 		gt_3dphoto[photo].human.insert(humanID, human)

# train_x = []
# train_y = []
# for i in range(57):
# 	for j in range(len(gt_cam1photo[i].human)):
# 		if j >= len(gt_3dphoto[i].human):
# 			break
# 		p = []
# 		#print (gt_cam1photo[1].human[0].cam2, gt_cam1photo[1].human[0].cam3)
# 		for k in range(10):
# 			p.append(gt_cam1photo[i].human[j].points[k].x)
# 			p.append(gt_cam1photo[i].human[j].points[k].y)
# 			p.append(gt_cam1photo[i].human[j].points[k].conf)
# 		human = gt_cam1photo[i].human[j].cam2
# 		for k in range(10):
# 			#print(human)
# 			p.append(human.points[k].x)
# 			p.append(human.points[k].y)
# 			p.append(human.points[k].conf)
# 		human = gt_cam1photo[i].human[j].cam3
# 		for k in range(10):
# 			p.append(human.points[k].x)
# 			p.append(human.points[k].y)
# 			p.append(human.points[k].conf)
# 		train_x.append(p)

# 		q = []
# 		for k in range(10):
# 			q.append(gt_3dphoto[i].human[j].points[k].x)
# 			q.append(gt_3dphoto[i].human[j].points[k].y)
# 			q.append(gt_3dphoto[i].human[j].points[k].z)
# 		train_y.append(q)
# print (train_x)
# print (train_y)





