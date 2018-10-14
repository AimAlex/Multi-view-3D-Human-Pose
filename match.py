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
		self.cam2 = 0
		self.cam3 = 0

	def init_cam1(self):
		points2 = [keyPoint(0, 0, 0) for i in range(10)]
		points3 = [keyPoint(0, 0, 0) for i in range(10)]
		self.cam2 = humanPoint(points2)
		self.cam3 = humanPoint(points3)

class photoHuman:
	def __init__(self):
		self.human = []

class matchHuman:
	def __init__(self, aa, bb, cc):
		self.cam1 = aa
		self.cam2 = bb
		self.cam3 = cc

class matchPhoto:
	def __init__(self):
		self.human = []

#ground truth
gdTruthFile = open("./data/camma_mvor_2018.json", "rb")

gdTruthJson = json.load(gdTruthFile)

cam3d = gdTruthJson["annotations3D"]


#2dpose
cam1File = open("./AlphaPose/cam1.json", "rb")
cam1 = json.load(cam1File)
cam2File = open("./AlphaPose/cam2.json", "rb")
cam2 = json.load(cam2File)
cam3File = open("./AlphaPose/cam3.json", "rb")
cam3 = json.load(cam3File)

#cam_info
camToRef = np.matrix([[-0.014959075298596,
        	-0.57090747056747,
        	0.82087811891685,
        	-1955.1484101018],
        	[-0.99444405304079,
        	0.094047813575387,
        	0.04728672259198,
        	880.95591882004],
        	[-0.10419813548242,
        	-0.8156099979843,
        	-0.56914240726732,
        	1470.139912924],
        	[0,
        	0,
        	0,
        	1]])
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

KRr2 = np.transpose(np.dot(np.dot(K1, np.transpose(R2)), t2))
KRr3 = np.transpose(np.dot(np.dot(K1, np.transpose(R3)), t3))

skew2 = np.array([[0, -KRr2[2], KRr2[1]], [KRr2[2], 0, -KRr2[0]], [-KRr2[1], KRr2[0], 0]])
skew3 = np.array([[0, -KRr3[2], KRr3[1]], [KRr3[2], 0, -KRr3[0]], [-KRr3[1], KRr3[0], 0]])

F2 = np.dot(np.dot(np.dot(K2.T.I, R2), K1.T), skew2)
F3 = np.dot(np.dot(np.dot(K3.T.I, R3), K1.T), skew3)

#print (F2)
#print (F3)


T1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
T2 = np.c_[R2,np.transpose([t2])]
T3 = np.c_[R3,np.transpose([t3])]

#calculate the point corresponding epipolar line
def findLine(x, y, num):
	#from second cam to first cam
	if num == 2:
		med = np.dot(K2.I, np.transpose([np.array([x, y, 1])]))
		xs = sp.symbols('xs:4')

		EQ1 = T2[0][0] * xs[0] + T2[0][1] * xs[1] + T2[0][2] * xs[2] + T2[0][3] - xs[3] * med[0]
		EQ2 = T2[1][0] * xs[0] + T2[1][1] * xs[1] + T2[1][2] * xs[2] + T2[1][3] - xs[3] * med[1]
		EQ3 = T2[2][0] * xs[0] + T2[2][1] * xs[1] + T2[2][2] * xs[2] + T2[2][3] - xs[3] * med[2]
		solution = sp.solve([EQ1, EQ2, EQ3], xs[:-1])
		#print(solution)

		med2 = sp.Matrix(np.dot(K1, np.transpose([np.array([xs[0], xs[1], xs[2]])])) / xs[2]).subs(solution)
		med2.subs(solution)
		#print(med2)

		ab = sp.symbols('ab:2')
		EQ5 = ab[1] - med2[1]
		EQ4 = (ab[0] - med2[0])
		solution2 = sp.solve([EQ4,EQ5], {ab[0], xs[3]})
		line = solution2[0][ab[0]]
		ploy = sp.Poly(line, ab[1])
		#print (a.coeffs())
		return ploy.coeffs()

	#from third cam to first cam
	if num == 3:
		med = np.dot(K3.I, np.transpose([np.array([x, y, 1])]))
		xs = sp.symbols('xs:4')

		EQ1 = T3[0][0] * xs[0] + T3[0][1] * xs[1] + T3[0][2] * xs[2] + T3[0][3] - xs[3] * med[0]
		EQ2 = T3[1][0] * xs[0] + T3[1][1] * xs[1] + T3[1][2] * xs[2] + T3[1][3] - xs[3] * med[1]
		EQ3 = T3[2][0] * xs[0] + T3[2][1] * xs[1] + T3[2][2] * xs[2] + T3[2][3] - xs[3] * med[2]
		solution = sp.solve([EQ1, EQ2, EQ3], xs[:-1])
		#print(solution)

		med2 = sp.Matrix(np.dot(K1, np.transpose([np.array([xs[0], xs[1], xs[2]])])) / xs[2]).subs(solution)
		med2.subs(solution)
		#print(med2)

		ab = sp.symbols('ab:2')
		EQ5 = ab[1] - med2[1]
		EQ4 = ab[0] - med2[0]
		solution2 = sp.solve([EQ4,EQ5], {ab[0], xs[3]})
		line = solution2[0][ab[0]]
		ploy = sp.Poly(line, ab[1])
		#print (a.coeffs())
		return ploy.coeffs()

#depth image
im_depth = cv.imread("./camma_mvor_dataset/day1/cam1/depth/000048.png", -1)

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
	human.init_cam1()
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

#transfor cam2 & cam3 to cam1

matchPhotos = [matchPhoto() for i in range(57)]

for i in range(57):
	photo1 = cam1photo[i]
	photo2 = cam2photo[i]
	photo3 = cam3photo[i]
	print(i)

	for human2 in photo2.human:
		mini = 10000
		minihuman = 0
		lines = []
		for j in range(10):
			#print(human2.points[j].x, human2.points[j].y)
			lines.append(findLine(human2.points[j].x, human2.points[j].y, 2))
		for human1 in photo1.human:
			dist = 0
			for j in range(10):
				dist += abs(human1.points[j].x - lines[j][0] * human1.points[j].y - lines[j][1]) / (1 + lines[j][0] * lines[j][0])
			if dist / 10 < mini:
				mini = dist / 10
				minihuman = human1
		if mini < 50:
			minihuman.cam2 = human2
			print(mini)

	for human3 in photo3.human:
		mini = 10000
		minihuman = 0
		lines = []
		for j in range(10):
			#print(human2.points[j].x, human2.points[j].y)
			lines.append(findLine(human3.points[j].x, human3.points[j].y, 3))
		for human1 in photo1.human:
			dist = 0
			for j in range(10):
				dist += abs(human1.points[j].x - lines[j][0] * human1.points[j].y - lines[j][1]) / (1 + lines[j][0] * lines[j][0])
			if dist / 10 < mini:
				mini = dist / 10
				minihuman = human1
		if mini < 50:
			minihuman.cam3 = human3
			print(mini)




#trans ref to cam1
X  = np.zeros(4)
for cam in cam3d:
	if cam["image_ids"] != "10010000048_10020000048_10030000048":
		continue
	for i in range(3):
		X[i] = cam["keypoints3D"][i]
	
X[3] = 1
#cam1True = np.dot(camToRef.I, np.transpose([X]))
#trans cam1 to 2d

Result = np.dot(np.dot(K1, T1), np.transpose([X]))
#print(Result)

k = np.zeros(3)
kk = np.zeros(3)
counter = 1
for cam in cam1:
	if cam["image_id"] != "000048.png" :
		continue
	k[0:2] = cam["keypoints"][0:2]
	break
k[2] = 1
s = im_depth[int(round(k[0]))][int(round(k[1]))]
#print (s * k)







