# hardwork
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


data_train=MNIST('./data',
                 download=True,
                 transform=transforms.Compose([transforms.Resize((32,32)),
                                               transforms.ToTensor()]))
data_test=MNIST('./data',
                train=False,
                download=True,
                transform=transforms.Compose([transforms.Resize((32, 32)),
                                              transforms.ToTensor()]))

data_train_loader=DataLoader(data_train,batch_size=256,shuffle=True)
data_test_loader=DataLoader(data_test,batch_size=1024,num_workers=8)


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(1,6,(3,3))
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,(3,3))
        self.pool2=nn.MaxPool2d(2,2)
        self.fc3=nn.Linear(16*16*6,120)
        self.fc4=nn.Linear(120,84)
        self.fc5=nn.Linear(84,10)

    def forward(self,x):
        x=self.pool1(torch.relu(self.conv1(x)))
        x=self.pool2(torch.relu(self.conv2(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc3(x))
        x=torch.relu(self.fc4(x))
        x=self.fc5(x)
        return x

model=LeNet()

model.train()
lr=0.01
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)

train_loss=0
correct=0
total=0

for batch_idx,(inputs,targets) in enumerate(data_train_loader):
    optimizer.zero_grad()
    outputs=model(inputs)
    loss=criterion(outputs,targets)
    loss.backward()
    optimizer.step()

    train_loss+=loss.item()
    _,predicted=outputs.max(1)
    total+=predicted.eq(targets).sum().item()

    print(batch_idx,len(trainloader),'Loss:%.3f|ACC:%.3f%%(%d/%d)')
