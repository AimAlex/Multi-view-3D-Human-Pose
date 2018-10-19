import json
import numpy as np
import cv2 as cv
import math
import sympy as sp

#define structure

class humanPoint:
	def __init__(self, Points):
		self.points = Points

class photoHuman:
	def __init__(self):
		self.human = []


day1cam1File = open("./AlphaPose/day1/cam1.json", "rb")
day1cam1 = json.load(day1cam1File)
day1cam2File = open("./AlphaPose/day1/cam2.json", "rb")
day1cam2 = json.load(day1cam2File)
day1cam3File = open("./AlphaPose/day1/cam3.json", "rb")
day1cam3 = json.load(day1cam3File)

day2cam1File = open("./AlphaPose/day2/cam1.json", "rb")
day2cam1 = json.load(day2cam1File)
day2cam2File = open("./AlphaPose/day2/cam2.json", "rb")
day2cam2 = json.load(day2cam2File)
day2cam3File = open("./AlphaPose/day2/cam3.json", "rb")
day2cam3 = json.load(day2cam3File)

day3cam1File = open("./AlphaPose/day3/cam1.json", "rb")
day3cam1 = json.load(day3cam1File)
day3cam2File = open("./AlphaPose/day3/cam2.json", "rb")
day3cam2 = json.load(day3cam2File)
day3cam3File = open("./AlphaPose/day3/cam3.json", "rb")
day3cam3 = json.load(day3cam3File)

day4cam1File = open("./AlphaPose/day4/cam1.json", "rb")
day4cam1 = json.load(day4cam1File)
day4cam2File = open("./AlphaPose/day4/cam2.json", "rb")
day4cam2 = json.load(day4cam2File)
day4cam3File = open("./AlphaPose/day4/cam3.json", "rb")
day4cam3 = json.load(day4cam3File)

#read day1
day1cam1photo = [photoHuman() for i in range(57)]
day1cam2photo = [photoHuman() for i in range(57)]
day1cam3photo = [photoHuman() for i in range(57)]
for cam in day1cam1:
	#print(cam["image_id"])
	no = int(cam["image_id"][3:6])
	keyList = cam["keypoints"]
	pointList = [(keyList[0] + keyList[3] + keyList[6])/3, (keyList[1] + keyList[4] + keyList[7])/3, (keyList[2] + keyList[5] + keyList[8])/3, (keyList[15] + keyList[18])/2, (keyList[16] + keyList[19])/2, (keyList[17] + keyList[20])/2, keyList[15], keyList[16], keyList[17], 
	keyList[18], keyList[19], keyList[20], keyList[33], keyList[34], keyList[35], keyList[36], keyList[37], keyList[38],
	keyList[21], keyList[22], keyList[23], keyList[24], keyList[25], keyList[26], keyList[27], keyList[28], keyList[29],
	keyList[30], keyList[31], keyList[32]]
	human = humanPoint(pointList)
	day1cam1photo[no].human.append(human)

for cam in day1cam2:
	#print(cam["image_id"])
	no = int(cam["image_id"][3:6])
	keyList = cam["keypoints"]
	pointList = [(keyList[0] + keyList[3] + keyList[6])/3, (keyList[1] + keyList[4] + keyList[7])/3, (keyList[2] + keyList[5] + keyList[8])/3, (keyList[15] + keyList[18])/2, (keyList[16] + keyList[19])/2, (keyList[17] + keyList[20])/2, keyList[15], keyList[16], keyList[17], 
	keyList[18], keyList[19], keyList[20], keyList[33], keyList[34], keyList[35], keyList[36], keyList[37], keyList[38],
	keyList[21], keyList[22], keyList[23], keyList[24], keyList[25], keyList[26], keyList[27], keyList[28], keyList[29],
	keyList[30], keyList[31], keyList[32]]
	human = humanPoint(pointList)
	day1cam2photo[no].human.append(human)

for cam in day1cam3:
	#print(cam["image_id"])
	no = int(cam["image_id"][3:6])
	keyList = cam["keypoints"]
	pointList = [(keyList[0] + keyList[3] + keyList[6])/3, (keyList[1] + keyList[4] + keyList[7])/3, (keyList[2] + keyList[5] + keyList[8])/3, (keyList[15] + keyList[18])/2, (keyList[16] + keyList[19])/2, (keyList[17] + keyList[20])/2, keyList[15], keyList[16], keyList[17], 
	keyList[18], keyList[19], keyList[20], keyList[33], keyList[34], keyList[35], keyList[36], keyList[37], keyList[38],
	keyList[21], keyList[22], keyList[23], keyList[24], keyList[25], keyList[26], keyList[27], keyList[28], keyList[29],
	keyList[30], keyList[31], keyList[32]]
	human = humanPoint(pointList)
	day1cam3photo[no].human.append(human)

#read day2
day2cam1photo = [photoHuman() for i in range(330)]
day2cam2photo = [photoHuman() for i in range(330)]
day2cam3photo = [photoHuman() for i in range(330)]
for cam in day2cam1:
	#print(cam["image_id"])
	no = int(cam["image_id"][3:6])
	keyList = cam["keypoints"]
	pointList = [(keyList[0] + keyList[3] + keyList[6])/3, (keyList[1] + keyList[4] + keyList[7])/3, (keyList[2] + keyList[5] + keyList[8])/3, (keyList[15] + keyList[18])/2, (keyList[16] + keyList[19])/2, (keyList[17] + keyList[20])/2, keyList[15], keyList[16], keyList[17], 
	keyList[18], keyList[19], keyList[20], keyList[33], keyList[34], keyList[35], keyList[36], keyList[37], keyList[38],
	keyList[21], keyList[22], keyList[23], keyList[24], keyList[25], keyList[26], keyList[27], keyList[28], keyList[29],
	keyList[30], keyList[31], keyList[32]]
	human = humanPoint(pointList)
	day2cam1photo[no].human.append(human)

for cam in day2cam2:
	#print(cam["image_id"])
	no = int(cam["image_id"][3:6])
	keyList = cam["keypoints"]
	pointList = [(keyList[0] + keyList[3] + keyList[6])/3, (keyList[1] + keyList[4] + keyList[7])/3, (keyList[2] + keyList[5] + keyList[8])/3, (keyList[15] + keyList[18])/2, (keyList[16] + keyList[19])/2, (keyList[17] + keyList[20])/2, keyList[15], keyList[16], keyList[17], 
	keyList[18], keyList[19], keyList[20], keyList[33], keyList[34], keyList[35], keyList[36], keyList[37], keyList[38],
	keyList[21], keyList[22], keyList[23], keyList[24], keyList[25], keyList[26], keyList[27], keyList[28], keyList[29],
	keyList[30], keyList[31], keyList[32]]
	human = humanPoint(pointList)
	day2cam2photo[no].human.append(human)

for cam in day2cam3:
	#print(cam["image_id"])
	no = int(cam["image_id"][3:6])
	keyList = cam["keypoints"]
	pointList = [(keyList[0] + keyList[3] + keyList[6])/3, (keyList[1] + keyList[4] + keyList[7])/3, (keyList[2] + keyList[5] + keyList[8])/3, (keyList[15] + keyList[18])/2, (keyList[16] + keyList[19])/2, (keyList[17] + keyList[20])/2, keyList[15], keyList[16], keyList[17], 
	keyList[18], keyList[19], keyList[20], keyList[33], keyList[34], keyList[35], keyList[36], keyList[37], keyList[38],
	keyList[21], keyList[22], keyList[23], keyList[24], keyList[25], keyList[26], keyList[27], keyList[28], keyList[29],
	keyList[30], keyList[31], keyList[32]]
	human = humanPoint(pointList)
	day2cam3photo[no].human.append(human)

#read day3
day3cam1photo = [photoHuman() for i in range(223)]
day3cam2photo = [photoHuman() for i in range(223)]
day3cam3photo = [photoHuman() for i in range(223)]
for cam in day3cam1:
	#print(cam["image_id"])
	no = int(cam["image_id"][3:6])
	keyList = cam["keypoints"]
	pointList = [(keyList[0] + keyList[3] + keyList[6])/3, (keyList[1] + keyList[4] + keyList[7])/3, (keyList[2] + keyList[5] + keyList[8])/3, (keyList[15] + keyList[18])/2, (keyList[16] + keyList[19])/2, (keyList[17] + keyList[20])/2, keyList[15], keyList[16], keyList[17], 
	keyList[18], keyList[19], keyList[20], keyList[33], keyList[34], keyList[35], keyList[36], keyList[37], keyList[38],
	keyList[21], keyList[22], keyList[23], keyList[24], keyList[25], keyList[26], keyList[27], keyList[28], keyList[29],
	keyList[30], keyList[31], keyList[32]]
	human = humanPoint(pointList)
	day3cam1photo[no].human.append(human)

for cam in day3cam2:
	#print(cam["image_id"])
	no = int(cam["image_id"][3:6])
	keyList = cam["keypoints"]
	pointList = [(keyList[0] + keyList[3] + keyList[6])/3, (keyList[1] + keyList[4] + keyList[7])/3, (keyList[2] + keyList[5] + keyList[8])/3, (keyList[15] + keyList[18])/2, (keyList[16] + keyList[19])/2, (keyList[17] + keyList[20])/2, keyList[15], keyList[16], keyList[17], 
	keyList[18], keyList[19], keyList[20], keyList[33], keyList[34], keyList[35], keyList[36], keyList[37], keyList[38],
	keyList[21], keyList[22], keyList[23], keyList[24], keyList[25], keyList[26], keyList[27], keyList[28], keyList[29],
	keyList[30], keyList[31], keyList[32]]
	human = humanPoint(pointList)
	day3cam2photo[no].human.append(human)

for cam in day3cam3:
	#print(cam["image_id"])
	no = int(cam["image_id"][3:6])
	keyList = cam["keypoints"]
	pointList = [(keyList[0] + keyList[3] + keyList[6])/3, (keyList[1] + keyList[4] + keyList[7])/3, (keyList[2] + keyList[5] + keyList[8])/3, (keyList[15] + keyList[18])/2, (keyList[16] + keyList[19])/2, (keyList[17] + keyList[20])/2, keyList[15], keyList[16], keyList[17], 
	keyList[18], keyList[19], keyList[20], keyList[33], keyList[34], keyList[35], keyList[36], keyList[37], keyList[38],
	keyList[21], keyList[22], keyList[23], keyList[24], keyList[25], keyList[26], keyList[27], keyList[28], keyList[29],
	keyList[30], keyList[31], keyList[32]]
	human = humanPoint(pointList)
	day3cam3photo[no].human.append(human)

#read day4
day4cam1photo = [photoHuman() for i in range(122)]
day4cam2photo = [photoHuman() for i in range(122)]
day4cam3photo = [photoHuman() for i in range(122)]
for cam in day4cam1:
	#print(cam["image_id"])
	no = int(cam["image_id"][3:6])
	keyList = cam["keypoints"]
	pointList = [(keyList[0] + keyList[3] + keyList[6])/3, (keyList[1] + keyList[4] + keyList[7])/3, (keyList[2] + keyList[5] + keyList[8])/3, (keyList[15] + keyList[18])/2, (keyList[16] + keyList[19])/2, (keyList[17] + keyList[20])/2, keyList[15], keyList[16], keyList[17], 
	keyList[18], keyList[19], keyList[20], keyList[33], keyList[34], keyList[35], keyList[36], keyList[37], keyList[38],
	keyList[21], keyList[22], keyList[23], keyList[24], keyList[25], keyList[26], keyList[27], keyList[28], keyList[29],
	keyList[30], keyList[31], keyList[32]]
	human = humanPoint(pointList)
	day4cam1photo[no].human.append(human)

for cam in day4cam2:
	#print(cam["image_id"])
	no = int(cam["image_id"][3:6])
	keyList = cam["keypoints"]
	pointList = [(keyList[0] + keyList[3] + keyList[6])/3, (keyList[1] + keyList[4] + keyList[7])/3, (keyList[2] + keyList[5] + keyList[8])/3, (keyList[15] + keyList[18])/2, (keyList[16] + keyList[19])/2, (keyList[17] + keyList[20])/2, keyList[15], keyList[16], keyList[17], 
	keyList[18], keyList[19], keyList[20], keyList[33], keyList[34], keyList[35], keyList[36], keyList[37], keyList[38],
	keyList[21], keyList[22], keyList[23], keyList[24], keyList[25], keyList[26], keyList[27], keyList[28], keyList[29],
	keyList[30], keyList[31], keyList[32]]
	human = humanPoint(pointList)
	day4cam2photo[no].human.append(human)

for cam in day4cam3:
	#print(cam["image_id"])
	no = int(cam["image_id"][3:6])
	keyList = cam["keypoints"]
	pointList = [(keyList[0] + keyList[3] + keyList[6])/3, (keyList[1] + keyList[4] + keyList[7])/3, (keyList[2] + keyList[5] + keyList[8])/3, (keyList[15] + keyList[18])/2, (keyList[16] + keyList[19])/2, (keyList[17] + keyList[20])/2, keyList[15], keyList[16], keyList[17], 
	keyList[18], keyList[19], keyList[20], keyList[33], keyList[34], keyList[35], keyList[36], keyList[37], keyList[38],
	keyList[21], keyList[22], keyList[23], keyList[24], keyList[25], keyList[26], keyList[27], keyList[28], keyList[29],
	keyList[30], keyList[31], keyList[32]]
	human = humanPoint(pointList)
	day4cam3photo[no].human.append(human)

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


with open('./AlphaPose/2dPose.json', 'w') as f:
    json.dump(humans, f)




