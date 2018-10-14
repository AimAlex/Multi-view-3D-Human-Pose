import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.fc1 = torch.nn.Linear(n_feature, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_hidden)
        self.fc3 = torch.nn.Linear(n_hidden, n_hidden)
        self.fc4 = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
    	x = F.relu(self.fc1(x))
    	x = F.relu(self.fc2(x))
    	x = F.relu(self.fc3(x))
    	x = F.relu(self.fc4(x))
    	return x

net = Net(30, 1024, 30)
net.cuda() 

train_loader = Data.DataLoader(dataset = train_data, batch_size = 512, shuffle = True)
optimizer =  torch.optim.Adam(net.parameters(), lr = 0.001)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

plt.ion()
plt.show()


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





