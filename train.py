import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt

class MultiLossFunc(nn.Module):
	def __init__(self):
		return

	def forward(self, [x1, x2, x3, x4], y):
		los = torch.nn.MSELoss()
		return (los(x1, y) + los(x2, y) + los(x3, y) + los(x4, y))

class MyDataset(Data.Dataset):
    def __init__(self, xx, yy):
        self.train_x = xx
        self.train_y = yy

    def __getitem__(self, index):#返回的是tensor
        b_x, b_y = self.train_x[index], self.train_y[index]
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

    def forward(self, x):
    	x1 = F.relu(self.fc1(x))
    	x2 = F.relu(self.fc2(x))
    	x3 = F.relu(self.fc3(x))
    	x4 = F.relu(self.fc4(x))
    	return [x1, x2, x_3, x_4]

net = Net(90, 1024, 30)
net.cuda()

dataset = MyDataset(train_x, train_y)
train_loader = Data.DataLoader(dataset, batch_size = 10, shuffle = True)
optimizer =  torch.optim.Adam(net.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
loss_function = MultiLossFunc()



for t in range(200):
	for step, (x, y) in enumerate(train_loader):
		b_x = x.cuda
		b_y = y.cuda

    	prediction = net(b_x)
    	loss = loss_function(prediction, b_y)
    	optimizer.zero_grad()
    	loss.backward()
    	optimizer.step()
    	scheduler.step()





