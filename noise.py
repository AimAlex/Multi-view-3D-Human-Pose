import json
import numpy as np
import cv2 as cv
import math
import sympy as sp

class keyPoint:
	def __init__(self,xx, yy, confidence):
		self.x = xx
		self.y = yy
		self.conf = confidence

class humanPoint:
	def __init__(self, Points):
		self.points = Points

class photoHuman:
	def __init__(self):
		self.human = []

#alphapose
cam1File = open("./AlphaPose/cam1.json", "rb")
cam1 = json.load(cam1File)
cam2File = open("./AlphaPose/cam2.json", "rb")
cam2 = json.load(cam2File)
cam3File = open("./AlphaPose/cam3.json", "rb")
cam3 = json.load(cam3File)

#read cam1
cam1photo = [photoHuman() for i in range(57)]

for cam in cam1:
	#print(cam["image_id"])
	no = int(cam["image_id"][4:6])
	keyList = cam["keypoints"]
	pointList = [1] * 10
	pointList[0] = keyPoint((keyList[0] + keyList[3] + keyList[6])/3, (keyList[1] + keyList[4] + keyList[7])/3, (keyList[2] + keyList[5] + keyList[8])/3)
	pointList[1] = keyPoint((keyList[15] + keyList[18])/2, (keyList[16] + keyList[19])/2, (keyList[17] + keyList[20])/2)
	pointList[2] = keyPoint(keyList[15], keyList[16], keyList[17])
	pointList[3] = keyPoint(keyList[18], keyList[19], keyList[20])
	pointList[4] = keyPoint(keyList[33], keyList[34], keyList[35])
	pointList[5] = keyPoint(keyList[36], keyList[37], keyList[38])
	pointList[6] = keyPoint(keyList[21], keyList[22], keyList[23])
	pointList[7] = keyPoint(keyList[24], keyList[25], keyList[26])
	pointList[8] = keyPoint(keyList[27], keyList[28], keyList[29])
	pointList[9] = keyPoint(keyList[30], keyList[31], keyList[32])
	human = humanPoint(pointList)
	cam1photo[no].human.append(human)
#print (cam1photo[47].human[0].points[0].x)

#read cam2
cam2photo = [photoHuman() for i in range(57)]
for cam in cam2:
	#print(cam["image_id"])
	no = int(cam["image_id"][4:6])
	keyList = cam["keypoints"]
	pointList = [1] * 10
	pointList[0] = keyPoint((keyList[0] + keyList[3] + keyList[6])/3, (keyList[1] + keyList[4] + keyList[7])/3, (keyList[2] + keyList[5] + keyList[8])/3)
	pointList[1] = keyPoint((keyList[15] + keyList[18])/2, (keyList[16] + keyList[19])/2, (keyList[17] + keyList[20])/2)
	pointList[2] = keyPoint(keyList[15], keyList[16], keyList[17])
	pointList[3] = keyPoint(keyList[18], keyList[19], keyList[20])
	pointList[4] = keyPoint(keyList[33], keyList[34], keyList[35])
	pointList[5] = keyPoint(keyList[36], keyList[37], keyList[38])
	pointList[6] = keyPoint(keyList[21], keyList[22], keyList[23])
	pointList[7] = keyPoint(keyList[24], keyList[25], keyList[26])
	pointList[8] = keyPoint(keyList[27], keyList[28], keyList[29])
	pointList[9] = keyPoint(keyList[30], keyList[31], keyList[32])
	human = humanPoint(pointList)
	cam2photo[no].human.append(human)

#read cam3
cam3photo = [photoHuman() for i in range(57)]
for cam in cam3:
	#print(cam["image_id"])
	no = int(cam["image_id"][4:6])
	keyList = cam["keypoints"]
	pointList = [1] * 10
	pointList[0] = keyPoint((keyList[0] + keyList[3] + keyList[6])/3, (keyList[1] + keyList[4] + keyList[7])/3, (keyList[2] + keyList[5] + keyList[8])/3)
	pointList[1] = keyPoint((keyList[15] + keyList[18])/2, (keyList[16] + keyList[19])/2, (keyList[17] + keyList[20])/2)
	pointList[2] = keyPoint(keyList[15], keyList[16], keyList[17])
	pointList[3] = keyPoint(keyList[18], keyList[19], keyList[20])
	pointList[4] = keyPoint(keyList[33], keyList[34], keyList[35])
	pointList[5] = keyPoint(keyList[36], keyList[37], keyList[38])
	pointList[6] = keyPoint(keyList[21], keyList[22], keyList[23])
	pointList[7] = keyPoint(keyList[24], keyList[25], keyList[26])
	pointList[8] = keyPoint(keyList[27], keyList[28], keyList[29])
	pointList[9] = keyPoint(keyList[30], keyList[31], keyList[32])
	human = humanPoint(pointList)
	cam3photo[no].human.append(human)

#ground truth
gdTruthFile = open("./data/camma_mvor_2018.json", "rb")
gdTruthJson = json.load(gdTruthFile)

cam2d = gdTruthJson["annotations"]

gt_cam1photo = [photoHuman() for i in range(57)]
gt_cam2photo = [photoHuman() for i in range(57)]
gt_cam3photo = [photoHuman() for i in range(57)]

for cam in cam2d:
	if int(cam["image_id"]/10**10) != 1:
		#print(int(cam["image_id"] / 10**10))
		continue
	view = int(cam["image_id"]/10**7) % 10
	photo = int(cam["image_id"]) % 1000
	pointList = [1] * 10
	keyList = cam["keypoints"]
	pointList[0] = keyPoint(keyList[0], keyList[1], keyList[2])
	pointList[1] = keyPoint(keyList[3], keyList[4], keyList[5])
	pointList[2] = keyPoint(keyList[6], keyList[7], keyList[8])
	pointList[3] = keyPoint(keyList[9], keyList[10], keyList[11])
	pointList[4] = keyPoint(keyList[12], keyList[13], keyList[14])
	pointList[5] = keyPoint(keyList[15], keyList[16], keyList[17])
	pointList[6] = keyPoint(keyList[18], keyList[19], keyList[20])
	pointList[7] = keyPoint(keyList[21], keyList[22], keyList[23])
	pointList[8] = keyPoint(keyList[24], keyList[25], keyList[26])
	pointList[9] = keyPoint(keyList[27], keyList[28], keyList[29])
	human = humanPoint(pointList)
	if view == 1:
		gt_cam1photo[photo].human.append(human)
	elif view == 2:
		gt_cam2photo[photo].human.append(human)
	elif view == 3:
		gt_cam3photo[photo].human.append(human)

#match 2dphoto with ground truth
counter = 0
totalNoise = 0
for i in range(57):
	for human in cam1photo[i].human:
		mini = 1000000
		mininum = 10
		for gt_human in gt_cam1photo[i].human:
			number = 10
			sum = 0
			for j in range(10):
				#print(gt_human.points[j].conf)
				if gt_human.points[j].conf != 2:
					number = number - 1
					continue
				#print ("ggg")
				sum += (gt_human.points[j].x - human.points[j].x) * (gt_human.points[j].x - human.points[j].x) + (gt_human.points[j].y - human.points[j].y) * (gt_human.points[j].y - human.points[j].y)
			if number == 0:
				continue
			if ((sum / number) < (mini / mininum)):
				mini = sum
				mininum = number
		if mini >= 10000:
			continue
		#print(mini, math.sqrt(mini / mininum), i)
		counter += mininum
		totalNoise += mini

	for human in cam2photo[i].human:
		mini = 1000000
		mininum = 10
		for gt_human in gt_cam2photo[i].human:
			number = 10
			sum = 0
			for j in range(10):
				if gt_human.points[j].conf != 2:
					number = number - 1
					continue
				sum += (gt_human.points[j].x - human.points[j].x) * (gt_human.points[j].x - human.points[j].x) + (gt_human.points[j].y - human.points[j].y) * (gt_human.points[j].y - human.points[j].y)
			if number == 0:
				continue
			if ((sum / number) < (mini / mininum)):
				mini = sum
				mininum = number
		if mini >= 10000:
			continue
		counter += mininum
		totalNoise += mini

	for human in cam3photo[i].human:
		mini = 1000000
		mininum = 10
		for gt_human in gt_cam3photo[i].human:
			number = 10
			sum = 0
			for j in range(10):
				if gt_human.points[j].conf != 2:
					number = number - 1
					continue
				sum += (gt_human.points[j].x - human.points[j].x) * (gt_human.points[j].x - human.points[j].x) + (gt_human.points[j].y - human.points[j].y) * (gt_human.points[j].y - human.points[j].y)
			#print(mininum, number)
			if number == 0:
				continue
			if ((sum / number) < (mini / mininum)):
				mini = sum
				mininum = number
		if mini >= 10000:
			continue
		counter += mininum
		totalNoise += mini

deviation = math.sqrt(totalNoise / counter)
print(deviation)















