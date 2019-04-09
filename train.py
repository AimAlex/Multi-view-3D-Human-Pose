import torch
import torch.nn as nn
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
import scipy.io as sio
from tensorboardX import SummaryWriter

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
deviation = 50
subdeviation = math.sqrt(deviation * deviation / 2)
torch.set_default_tensor_type('torch.DoubleTensor')
ave_x = 97 + 49
ave_y = -573 + 127
ave_z = 5108 - 2398
batchSize = 512
#tensor board
writer = SummaryWriter('./tensorboard')

def GaussianConf(wx, wy):
	w = math.sqrt(wx * wx + wy * wy)
	c = 1 - w / (4 * deviation)
	#print (max(c, 0))
	return max(c, 0)

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

#ground truth mvor
# gdTruthFile = open("./data/camma_mvor_2018.json", "rb")
# gdTruthJson = json.load(gdTruthFile)

#read 3d ground truth
#cam3d = gdTruthJson["annotations3D"]

# gt_3dhuman = []

# test_x = []
# test_y = []

#read test_data on mvor
# for cam in cam3d:
# 	keyList = cam["keypoints3D"]
# 	p = []
# 	q = []
# 	for i in range(10):
# 		p.append(keyList[4 * i])
# 		p.append(keyList[4 * i + 1])
# 		p.append(keyList[4 * i + 2])
# 		point2d = cam2d(keyList[4 * i], keyList[4 * i + 1], keyList[4 * i + 2])
# 		q.append(point2d[0])
# 		q.append(point2d[1])
# 		q.append(0)
# 		q.append(point2d[2])
# 		q.append(point2d[3])
# 		q.append(0)
# 		q.append(point2d[4])
# 		q.append(point2d[5])
# 		q.append(0)
# 		#print(point2d[0])
# 	test_y.append(np.array(p))
# 	test_x.append(np.array(q))

#read human3.6M
train36_file = sio.loadmat("./human3.6/train_3d.mat")
test36_file = sio.loadmat("./human3.6/val_3d.mat")
train36 = train36_file['data']
test36 = test36_file['data']
#print(test36[0][0][0].shape)

train_x = []
train_y = []
# all_x = 0
# num = 0
for human in train36:
	human36 = human[0][0]
	p = []
	q = []
	# all_x += human36[9][2] + human36[8][2] + human36[10][2] + human36[13][2] + human36[3][2] + human36[0][2] + human36[11][2] + human36[14][2] + human36[12][2] + human36[15][2]
	#print (human36.shape)
	# num += 10
	#head
	p.append(human36[9][0] - ave_x)
	p.append(human36[9][1] - ave_y)
	p.append(human36[9][2] - ave_z)
	point2d = cam2d(human36[9][0] - ave_x, human36[9][1] - ave_y, human36[9][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(0)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(0)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(0)

	#neck
	p.append(human36[8][0] - ave_x)
	p.append(human36[8][1] - ave_y)
	p.append(human36[8][2] - ave_z)
	point2d = cam2d(human36[8][0] - ave_x, human36[8][1] - ave_y, human36[8][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(0)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(0)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(0)

	#left shoulder

	p.append(human36[10][0] - ave_x)
	p.append(human36[10][1] - ave_y)
	p.append(human36[10][2] - ave_z)
	point2d = cam2d(human36[10][0] - ave_x, human36[10][1] - ave_y, human36[10][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(0)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(0)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(0)

	#right shoulder 

	p.append(human36[13][0] - ave_x)
	p.append(human36[13][1] - ave_y)
	p.append(human36[13][2] - ave_z)
	point2d = cam2d(human36[13][0] - ave_x, human36[13][1] - ave_y, human36[13][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(0)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(0)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(0)

	#left hip

	p.append(human36[3][0] - ave_x)
	p.append(human36[3][1] - ave_y)
	p.append(human36[3][2] - ave_z)
	point2d = cam2d(human36[3][0] - ave_x, human36[3][1] - ave_y, human36[3][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(0)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(0)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(0)

	#right hip

	p.append(human36[0][0] - ave_x)
	p.append(human36[0][1] - ave_y)
	p.append(human36[0][2] - ave_z)
	point2d = cam2d(human36[0][0] - ave_x, human36[0][1] - ave_y, human36[0][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(0)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(0)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(0)

	#left elbow

	p.append(human36[11][0] - ave_x)
	p.append(human36[11][1] - ave_y)
	p.append(human36[11][2] - ave_z)
	point2d = cam2d(human36[11][0] - ave_x, human36[11][1] - ave_y, human36[11][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(0)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(0)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(0)

	#right elbow

	p.append(human36[14][0] - ave_x)
	p.append(human36[14][1] - ave_y)
	p.append(human36[14][2] - ave_z)
	point2d = cam2d(human36[14][0] - ave_x, human36[14][1] - ave_y, human36[14][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(0)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(0)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(0)

	#left wrist

	p.append(human36[12][0] - ave_x)
	p.append(human36[12][1] - ave_y)
	p.append(human36[12][2] - ave_z)
	point2d = cam2d(human36[12][0] - ave_x, human36[12][1] - ave_y, human36[12][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(0)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(0)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(0)

	#right wrist

	p.append(human36[15][0] - ave_x)
	p.append(human36[15][1] - ave_y)
	p.append(human36[15][2] - ave_z)
	point2d = cam2d(human36[15][0] - ave_x, human36[15][1] - ave_y, human36[15][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(0)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(0)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(0)


	train_y.append(np.array(p))
	train_x.append(np.array(q))
train_x = torch.from_numpy(np.array(train_x).astype(np.double)).double()
train_y = torch.from_numpy(np.array(train_y).astype(np.double)).double()

valid_x = []
valid_y = []
for human in test36:
	human36 = human[0][0]
	#print (human36.shape)
	p = []
	q = []

	#head
	p.append(human36[9][0] - ave_x)
	p.append(human36[9][1] - ave_y)
	p.append(human36[9][2] - ave_z)
	point2d = cam2d(human36[9][0] - ave_x, human36[9][1] - ave_y, human36[9][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(1)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(1)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(1)

	#neck
	p.append(human36[8][0] - ave_x)
	p.append(human36[8][1] - ave_y)
	p.append(human36[8][2] - ave_z)
	point2d = cam2d(human36[8][0] - ave_x, human36[8][1] - ave_y, human36[8][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(1)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(1)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(1)

	#left shoulder

	p.append(human36[10][0] - ave_x)
	p.append(human36[10][1] - ave_y)
	p.append(human36[10][2] - ave_z)
	point2d = cam2d(human36[10][0] - ave_x, human36[10][1] - ave_y, human36[10][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(1)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(1)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(1)

	#right shoulder 

	p.append(human36[13][0] - ave_x)
	p.append(human36[13][1] - ave_y)
	p.append(human36[13][2] - ave_z)
	point2d = cam2d(human36[13][0] - ave_x, human36[13][1] - ave_y, human36[13][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(1)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(1)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(1)

	#left hip

	p.append(human36[3][0] - ave_x)
	p.append(human36[3][1] - ave_y)
	p.append(human36[3][2] - ave_z)
	point2d = cam2d(human36[3][0] - ave_x, human36[3][1] - ave_y, human36[3][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(1)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(1)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(1)

	#right hip

	p.append(human36[0][0] - ave_x)
	p.append(human36[0][1] - ave_y)
	p.append(human36[0][2] - ave_z)
	point2d = cam2d(human36[0][0] - ave_x, human36[0][1] - ave_y, human36[0][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(1)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(1)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(1)

	#left elbow

	p.append(human36[11][0] - ave_x)
	p.append(human36[11][1] - ave_y)
	p.append(human36[11][2] - ave_z)
	point2d = cam2d(human36[11][0] - ave_x, human36[11][1] - ave_y, human36[11][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(1)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(1)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(1)

	#right elbow

	p.append(human36[14][0] - ave_x)
	p.append(human36[14][1] - ave_y)
	p.append(human36[14][2] - ave_z)
	point2d = cam2d(human36[14][0] - ave_x, human36[14][1] - ave_y, human36[14][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(1)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(1)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(1)

	#left wrist

	p.append(human36[12][0] - ave_x)
	p.append(human36[12][1] - ave_y)
	p.append(human36[12][2] - ave_z)
	point2d = cam2d(human36[12][0] - ave_x, human36[12][1] - ave_y, human36[12][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(1)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(1)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(1)

	#right wrist

	p.append(human36[15][0] - ave_x)
	p.append(human36[15][1] - ave_y)
	p.append(human36[15][2] - ave_z)
	point2d = cam2d(human36[15][0] - ave_x, human36[15][1] - ave_y, human36[15][2] - ave_z)
	q.append(point2d[0])
	q.append(point2d[1])
	q.append(1)
	q.append(point2d[2])
	q.append(point2d[3])
	q.append(1)
	q.append(point2d[4])
	q.append(point2d[5])
	q.append(1)


	valid_y.append(np.array(p))
	valid_x.append(np.array(q))
valid_x = torch.from_numpy(np.array(valid_x).astype(np.double)).double()
valid_y = torch.from_numpy(np.array(valid_y).astype(np.double)).double()

# print (all_x/num)
class MultiLossFunc(torch.nn.Module):
	def __init__(self):
		super(MultiLossFunc, self).__init__()
		return

	def forward(self, x1, x2, x3, y):
		#print (x1[0].shape)
		scale = len(x1)
		loss = torch.pow(F.pairwise_distance(x1, y), 2) + torch.pow(F.pairwise_distance(x2, y), 2) + torch.pow(F.pairwise_distance(x3, y), 2)
		#print (loss.shape)
		los = loss[0]
		for i in range(scale - 1):
			los += loss[i + 1]
		#print (los)

		return los / scale


class MyDataset(Data.Dataset):
    def __init__(self, xx, yy):
        self.train_x = xx
        self.train_y = yy

    def __getitem__(self, index):
    	#print(index)
    	for i in range(30):
    		wx = random.gauss(0, subdeviation)
    		wy = random.gauss(0, subdeviation)
    		self.train_x[index][3 * i] += wx
    		self.train_x[index][3 * i + 1] += wy
    		self.train_x[index][3 * i + 2] = GaussianConf(wx, wy)
    	# print(self.train_x[index][3 * i + 2])
    	b_x, b_y = self.train_x[index], self.train_y[index]
    	#print(b_y)
    	return b_x, b_y

    def __len__(self):
        return len(self.train_x)

class Net(torch.nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.fc1 = nn.Sequential(nn.Linear(n_feature, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(True))
		self.fc2 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(True))
		self.fc3 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(True))
		self.fc4 = nn.Linear(n_hidden, n_output)

		self.fc5 = nn.Sequential(nn.Linear(n_output + n_feature, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(True))
		self.fc6 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(True))
		self.fc7 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(True))
		self.fc8 = torch.nn.Linear(n_hidden, n_output)

		self.fc9 = nn.Sequential(nn.Linear(n_output + n_feature, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(True))
		self.fc10 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(True))
		self.fc11 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(True))
		self.fc12 = torch.nn.Linear(n_hidden, n_output)

	def forward(self, x):
		x1 = self.fc1(x)
		x1 = self.fc2(x1)
		x1 = self.fc3(x1)
		x1 = self.fc4(x1)

		x2 = torch.cat((x1, x), 1)
		x2 = self.fc5(x2)
		x2 = self.fc6(x2)
		x2 = self.fc7(x2)
		x2 = self.fc8(x2)

		x3 = torch.cat((x2, x), 1)
		x3 = self.fc9(x3)
		x3 = self.fc10(x3)
		x3 = self.fc11(x3)
		x3 = self.fc12(x3)

		return x1, x2, x3

net = Net(90, 1024, 30)
#print(net)
net.cuda()

# train_sum = len(train_x)

# test_x = train_x[int(0.9 * train_sum): train_sum]
test_x_save = train_x
# test_y = train_y[int(0.9 * train_sum): train_sum]

dataset = MyDataset(train_x, train_y)
validset = MyDataset(valid_x, valid_y)
train_loader = Data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers=2)
valid_loader = Data.DataLoader(validset, batch_size = 1024, shuffle = False, num_workers=2)
optimizer =  torch.optim.Adam(net.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.97)
loss_function = MultiLossFunc()
running_loss = 0.0
for t in range(200):
	print("epoch:", t)
    
    #train
	for step, (x, y) in enumerate(train_loader):
		b_x = x.cuda()
		b_y = y.cuda()

		#print(b_x)
		prediction = net(b_x)
		loss = loss_function(prediction[0], prediction[1], prediction[2], b_y.double())
		running_loss += math.sqrt(loss.cpu().data.numpy() / 30)
		if(step % 50 == 49):
			writer.add_scalar('accuracy', running_loss / 50, t * 1200 * 256 / batchSize + step)
			print('[%d, %5d] loss: %.3f' % (t+1, step+1, running_loss / 50))
			running_loss = 0

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()

	correct = 0
	total = 0
	net.eval()
	for step, (x, y) in enumerate(valid_loader):
		b_x = x.cuda()
		b_y = y.cuda()
		#print(b_x)
		prediction = net(b_x)
		loss = torch.pow(F.pairwise_distance(prediction[2], b_y), 2) / 10
		total += 1
		correct += math.sqrt(torch.mean(loss).cpu().data.numpy())
	print ("*************test: ", correct / (total))
	net.train()
#     num = len(test_y)
# 	#print(num)
#     a_x = [0 for i in range(num)]
#     for j in range(num):
#         for i in range(30):
# 			wx = random.gauss(0, subdeviation)
# 			wy = random.gauss(0, subdeviation)
# 			test_x[j][3 * i] = test_x_save[j][3 * i] + wx
# 			test_x[j][3 * i + 1] = test_x_save[j][3 * i + 1] + wy
# 			test_x[j][3 * i + 2] = GaussianConf(wx, wy)
# 			a_x[j] = torch.from_numpy(test_x[j].astype(np.double)).double().cuda()

# 	los = 0

# 	for i in range(num):
# 		test_output = net(a_x[i])
# 		pre_y = test_output[2].cpu().data.numpy()
# 		for j in range(30):
# 			los += (pre_y[j] - test_y[i][j]) * (pre_y[j] - test_y[i][j])
# 			#print("test :", pre_y[j], test_y[i][j])
# 	print ("*************test: ", math.sqrt(los / (10 * num)))
writer.close()
torch.save(net, 'model_ao.pkl')






