import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import json
import numpy as np
import cv2 as cv
import math
import sympy as sp
import random

deviation = 4
subdeviation = math.sqrt(deviation * deviation / 2)
torch.set_default_tensor_type('torch.DoubleTensor')

def GaussianConf(wx, wy):
	w = math.sqrt(wx * wx + wy * wy)
	c = 1 - w / (8 * deviation)
	#print (max(c, 0))
	return max(c, 0)

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

T1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
T2 = np.c_[R2,np.transpose([t2])]
T3 = np.c_[R3,np.transpose([t3])]

#calculate the 2d ground truth
def cam2d(xx, yy, zz):
	if int(xx) == 0 & int(yy) == 0 & int(zz) == 0:
		return 0, 0, 0, 0, 0, 0
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

	return pp1[0], pp1[1], pp2[0], pp2[1], pp3[0], pp3[1]

#ground truth
gdTruthFile = open("./data/camma_mvor_2018.json", "rb")
gdTruthJson = json.load(gdTruthFile)

#read 3d ground truth
cam3d = gdTruthJson["annotations3D"]

gt_3dhuman = []

train_x = []
train_y = []

for cam in cam3d:
#read train_y
	keyList = cam["keypoints3D"]
	p = []
	q = []
	for i in range(10):
		p.append(keyList[4 * i])
		p.append(keyList[4 * i + 1])
		p.append(keyList[4 * i + 2])
		point2d = cam2d(keyList[4 * i], keyList[4 * i + 1], keyList[4 * i + 2])
		q.append(point2d[0])
		q.append(point2d[1])
		q.append(0)
		q.append(point2d[2])
		q.append(point2d[3])
		q.append(0)
		q.append(point2d[4])
		q.append(point2d[5])
		q.append(0)
		#print(point2d[0])
	train_y.append(np.array(p))
	train_x.append(np.array(q))

class MultiLossFunc(torch.nn.Module):
	def __init__(self):
		super(MultiLossFunc, self).__init__()
		return

	def forward(self, x1, x2, x3, y):
		los = torch.nn.MSELoss()
		return (los(x1, y) + los(x2, y) + los(x3, y))

class MyDataset(Data.Dataset):
    def __init__(self, xx, yy):
        self.train_x = xx
        self.train_y = yy

    def __getitem__(self, index):
    	#print(index)
    	self.train_x[index]
    	self.train_y[index]
    	for i in range(30):
    		wx = random.gauss(0, subdeviation)
    		wy = random.gauss(0, subdeviation)
    		self.train_x[index][3 * i] += wx
    		self.train_x[index][3 * i + 1] += wy
    		self.train_x[index][3 * i + 2] = GaussianConf(wx, wy)
    	#print (self.train_x[index].shape)
    	b_x, b_y = torch.from_numpy(self.train_x[index].astype(np.double)).double(), torch.from_numpy(self.train_y[index].astype(np.double)).double()
    	#print(b_y)
    	return b_x, b_y

    def __len__(self):
        return len(self.train_x)

class Net(torch.nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.fc1 = torch.nn.Linear(n_feature, n_hidden)
		self.fc2 = torch.nn.Linear(n_hidden, n_hidden)
		self.fc3 = torch.nn.Linear(n_hidden, n_hidden)
		self.fc4 = torch.nn.Linear(n_hidden, n_output)

		self.fc5 = torch.nn.Linear(n_output, n_hidden)
		self.fc6 = torch.nn.Linear(n_hidden, n_hidden)
		self.fc7 = torch.nn.Linear(n_hidden, n_hidden)
		self.fc8 = torch.nn.Linear(n_hidden, n_output)

		self.fc9 = torch.nn.Linear(n_output, n_hidden)
		self.fc10 = torch.nn.Linear(n_hidden, n_hidden)
		self.fc11 = torch.nn.Linear(n_hidden, n_hidden)
		self.fc12 = torch.nn.Linear(n_hidden, n_output)

	def forward(self, x):
		x1 = F.relu(self.fc1(x))
		x1 = F.relu(self.fc2(x1))
		x1 = F.relu(self.fc3(x1))
		x1 = self.fc4(x1)

		x2 = F.relu(self.fc5(x1))
		x2 = F.relu(self.fc6(x2))
		x2 = F.relu(self.fc7(x2))
		x2 = self.fc8(x2)

		x3 = F.relu(self.fc9(x2))
		x3 = F.relu(self.fc10(x3))
		x3 = F.relu(self.fc11(x3))
		x3 = self.fc12(x3)

		return x1, x2, x3

net = Net(90, 1024, 30)
#print(net)
#net.cuda()

train_sum = len(train_x)

test_x = train_x[int(0.9 * train_sum): train_sum]
test_x_save = train_x[int(0.9 * train_sum): train_sum]
test_y = train_y[int(0.9 * train_sum): train_sum]

dataset = MyDataset(train_x[: int(0.9 * train_sum)], train_y[: int(0.9 * train_sum)])
train_loader = Data.DataLoader(dataset, batch_size = 512, shuffle = True)
optimizer =  torch.optim.Adam(net.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
loss_function = MultiLossFunc()

for t in range(256):
	print("epoch:", t)

	for step, (x, y) in enumerate(train_loader):
		b_x = x#.cuda
		b_y = y#.cuda

		#print(b_x)
		prediction = net(b_x)
		loss = loss_function(prediction[0], prediction[1], prediction[2], b_y.double())
		print(math.sqrt(loss.data.numpy() / 90))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()

		if(step % 100 == 0):
			num = len(test_y)
			#print(num)
			a_x = [0 for i in range(num)]
			for j in range(num):
				for i in range(30):
					wx = random.gauss(0, subdeviation)
					wy = random.gauss(0, subdeviation)
					test_x[j][3 * i] = test_x_save[j][3 * i] + wx
					test_x[j][3 * i + 1] = test_x_save[j][3 * i + 1] + wy
					test_x[j][3 * i + 2] = GaussianConf(wx, wy)
				a_x[j] = torch.from_numpy(test_x[j].astype(np.double)).double()

			los = 0

			for i in range(num):
				test_output = net(a_x[i])
				pre_y = test_output[2].data.numpy()
				for j in range(30):
					los += (pre_y[j] - test_y[i][j]) * (pre_y[j] - test_y[i][j])
					#print(pre_y[j], b_y[i][j])
			print (math.sqrt(los / (30 * num)))








