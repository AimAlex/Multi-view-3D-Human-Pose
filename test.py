import pickle
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
import random

deviation = 10
subdeviation = math.sqrt(deviation * deviation / 2)

file = open('match_result.pkl', 'rb')
camList = pickle.load(file)

def GaussianConf(wx, wy):
    w = math.sqrt(wx * wx + wy * wy)
    c = 1 - w / (4 * deviation)
    #print (max(c, 0))
    return max(c, 0)

camToRef = np.matrix([[-0.014959075298595706,
        -0.570907470567468,
        0.8208781189168504,
        -1955.1484101018448],
        [-0.9944440530407923,
        0.09404781357538733,
        0.04728672259197974,
        880.9559188200409],
        [-0.10419813548241928,
        -0.815609997984297,
        -0.5691424072673225,
        1470.1399129239926],
        [0, 0, 0, 1]])
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
    pp1 = np.array(np.dot(K1, cc1) / cc1[2])

    #cam2
    cc2 = np.dot(T2, np.transpose([point]))
    pp2 = np.array(np.dot(K2, cc2) / cc2[2])

    #cam3
    cc3 = np.dot(T3, np.transpose([point]))
    pp3 = np.array(np.dot(K3, cc3) / cc3[2])
    
    #print(pp1)
    return pp1[0][0], pp1[1][0], pp2[0][0], pp2[1][0], pp3[0][0], pp3[1][0]

def camRef(xx, yy, zz):
    if int(xx) == 0 & int(yy) == 0 & int(zz) == 0:
        return 0, 0, 0
    point = np.array([xx, yy, zz, 1])
    human = np.array(np.dot(camToRef, point))
    # print(human.shape)
    return human[0][0], human[0][1], human[0][2]

def constrain(x):
    if x >= 1:
        return 1
    else:
        return 0
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

net = torch.load('model66.pkl')
net.cuda()
net.eval()
class MyDataset(Data.Dataset):
    def __init__(self, xx, yy):
        self.train_x = np.array(xx)
        self.train_y = np.array(yy)
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

gdTruthFile = open("./data/camma_mvor_2018.json", "rb")
gdTruthJson = json.load(gdTruthFile)

cam3d = gdTruthJson["annotations3D"]
test_x = []
test_y = []
# all_x = 0
# num = 0
for num, cam in enumerate(cam3d):
    points = cam["keypoints3D"]
    human3d = []
    human2d = []
    for i in range(10):
    # x, y, z = camRef(points[4 * i], points[4 * i + 1], points[4 * i + 2]);
        x = points[4 * i]
        y = points[4 * i + 1]
        z = points[4 * i + 2]
    # if (z != 0) :
    #     all_x += z
    #     num += 1
        point2d = cam2d(x, y ,z);
    # print(point2d[0])
        human3d.append(x);
        human3d.append(y);
        human3d.append(z);
        human2d.append(point2d[0])
        human2d.append(point2d[1])
        human2d.append(constrain(points[4 * i + 3]))
        human2d.append(point2d[2])
        human2d.append(point2d[3])
        human2d.append(constrain(points[4 * i + 3]))
        human2d.append(point2d[4])
        human2d.append(point2d[5])
        human2d.append(constrain(points[4 * i + 3]))
    test_x.append(human2d)
    test_y.append(human3d)
    # print (human2d)
    # print (human3d)
    # if num > 4:
    #     exit()

test_data = MyDataset(test_x, test_y)
test_loader = Data.DataLoader(test_data, batch_size = 4, shuffle = False, num_workers = 2)
allcorrect = 0
total = 0
for step, (x, y) in enumerate(test_loader):
    b_x = x.cuda()
    b_y = y.cuda()
    prediction = net(b_x)
    loss = torch.pow(F.pairwise_distance(prediction[2], b_y), 2) / 10
    correct = math.sqrt(torch.mean(loss).cpu().data.numpy())
    allcorrect += correct
    total += 1
    print('Step %d , loss : %.3f' % (step, correct))
print ("*************test: ", allcorrect / (total))

# print (all_x / num)
# for day, item in enumerate(camList):
#     print ("time: ", day)
#     for no, human in enumerate(item):
#     print ("human: ", no + 1)
#     x = torch.from_numpy(np.array(human))
#     xx = x.cuda()
#     prediction = net(xx)[2]
#     print(prediction.cpu().data.numpy())

person3D = {}
for cam in cam3d:
    name = cam["image_ids"]
    person = cam["person_id"]
    day = int(name[0])
    time = int(name[8: 11])
    person3D[(day, time, person)] = cam["keypoints3D"]

cam_2d = gdTruthJson["annotations"]
person2D_1 = {}
person2D_2 = {}
person2D_3 = {}

for cam in cam_2d:
    name = str(cam["image_id"])
    view = int(name[3])
    day = int(name[0])
    time = int(name[8: 11])
    person = cam["person_id"]
    if view == 1:
        person2D_1[(day, time, person)] = cam["keypoints"]
    elif view == 2:
        person2D_2[(day, time, person)] = cam["keypoints"]
    elif view == 3:
        person2D_3[(day, time, person)] = cam["keypoints"]

gd_x = []
gd_y = []
# k = (1, 25, 0)
# print (person2D_1[k])
# print (person2D_2[k])
# print (person2D_3[k])
# print (person3D[k])
dev1 = 0
dev2 = 0
dev3 = 0
count1 = 0
count2 = 0
count3 = 0
for (k, v) in person3D.items():
    if k not in person2D_1:
        view_1 = [0] * 30
    else:
        view_1 = person2D_1[k]
    if k not in person2D_2:
        view_2 = [0] * 30
    else:
        view_2 = person2D_2[k]
    if k not in person2D_3:
        view_3 = [0] * 30
    else:
        view_3 = person2D_3[k]
    human3d = []
    human2d = []
    for i in range(10):
        human3d.append(v[4 * i])
        human3d.append(v[4 * i + 1])
        human3d.append(v[4 * i + 2])

        human2d.append(view_1[3 * i])
        human2d.append(view_1[3 * i + 1])
        human2d.append(constrain(view_1[3 * i + 2]))

        human2d.append(view_2[3 * i])
        human2d.append(view_2[3 * i + 1])
        human2d.append(constrain(view_2[3 * i + 2]))

        human2d.append(view_3[3 * i])
        human2d.append(view_3[3 * i + 1])
        human2d.append(constrain(view_3[3 * i + 2]))

        x = v[4 * i]
        y = v[4 * i + 1]
        z = v[4 * i + 2]
        point2d = cam2d(x, y ,z)
        if view_1[3 * i] != 0:
            dev1 += math.sqrt((view_1[3 * i] - point2d[0]) * (view_1[3 * i] - point2d[0]) + (view_1[3 * i + 1] - point2d[1]) * (view_1[3 * i + 1] - point2d[1]))
            count1 += 1
        if view_2[3 * i] != 0:
            dev2 += math.sqrt((view_2[3 * i] - point2d[2]) * (view_2[3 * i] - point2d[2]) + (view_2[3 * i + 1] - point2d[3]) * (view_2[3 * i + 1] - point2d[3]))
            count2 += 1
        if view_3[3 * i] != 0:
            dev3 += math.sqrt((view_3[3 * i] - point2d[4]) * (view_3[3 * i] - point2d[4]) + (view_3[3 * i + 1] - point2d[5]) * (view_3[3 * i + 1] - point2d[5]))
            count3 += 1
    gd_x.append(human2d)
    gd_y.append(human3d)
    # if k == (1, 25, 0):
    #     print (human2d)
    #     print (human3d)

print ("projection loss : view1 : %.3f, view2 : %.3f, view3 : %.3f" % (dev1 / count1, dev2 / count2, dev3 /count3))
gd_data = MyDataset(gd_x, gd_y)
gd_loader = Data.DataLoader(gd_data, batch_size = 1, shuffle = False, num_workers = 2)
allcorrect = 0
total = 0
for step, (x, y) in enumerate(gd_loader):
    b_x = x.cuda()
    b_y = y.cuda()
    prediction = net(b_x)
    loss = torch.pow(F.pairwise_distance(prediction[2], b_y), 2) / 10
    correct = math.sqrt(torch.mean(loss).cpu().data.numpy())
    allcorrect += correct
    total += 1
    print('Step %d , loss : %.3f' % (step, correct))
print ("*************test: ", allcorrect / (total))

# cam_2d = gdTruthJson["annotations"]
# cam1human = 0
# cam2human = 0
# cam3human = 0
# for cam in cam_2d:
#     name = cam["image_id"]
#     #print (type(name))
#     if cam["person_id"] != 1:
#     continue
#     if name == 10010000044:
#     cam1human = cam['keypoints']
#     print (cam["person_id"])
#     elif name == 10020000044:
#     cam2human = cam['keypoints']
#     print (cam["person_id"])
#     elif name == 10030000044:
#     cam3human = cam['keypoints']
#     print (cam["person_id"])
# pointList = []
# print(type(cam1human))
# for i in range(10):
#     pointList.append(cam1human[3 * i])
#     pointList.append(cam1human[3 * i + 1])
#     pointList.append(constrain(cam1human[3 * i + 2]))
#     pointList.append(cam2human[3 * i])
#     pointList.append(cam2human[3 * i + 1])
#     pointList.append(constrain(cam2human[3 * i + 2]))
#     pointList.append(cam3human[3 * i])
#     pointList.append(cam3human[3 * i + 1])
#     pointList.append(constrain(cam3human[3 * i + 2]))
# print ("human: ", 44)
# x = torch.from_numpy(np.array(pointList))
# xx = x.cuda()
# prediction = net(xx)[2]
# print(prediction.cpu().data.numpy())