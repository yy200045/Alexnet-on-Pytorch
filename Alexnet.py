import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transformss
import torch.optim as optim
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os

#超参数
writer = SummaryWriter('./logs')
device = torch.device("cuda:0")
batchsize = 32
Epochs = 10
modelpath = './model/model.pkl'
#载入数据
transform = transformss.Compose(
    [transformss.ToTensor(),
     transformss.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#数据增强
transform1 = transformss.Compose(
    [
        transformss.RandomHorizontalFlip(p=0.5),
        transformss.RandomVerticalFlip(p=0.5),
        transformss.RandomGrayscale(),
        transformss.ToTensor(),
        transformss.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform1)
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False,transform=transform)
trainloader = torch.utils.data.DataLoader(cifar_train, batch_size=batchsize, shuffle=True)
testloader = torch.utils.data.DataLoader(cifar_test, batch_size=batchsize, shuffle=True)


#定义神经网络
class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0)
        )
        #全连接层
        self.dense = nn.Sequential(
            nn.Linear(128,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = x.view(-1, 128)
            x = self.dense(x)
            return x


net = Alexnet().to(device)


def Accuracy():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _,  predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    #print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return 100.0*correct/total


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def Accuracy1():
    correct = 0
    total = 0
    flag = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _,  predicted = torch.max(outputs.data, 1)

            if flag == 0:
                images = images.cuda().data.cpu()
                imshow(torchvision.utils.make_grid(images))
                print('Label: ', ' '.join('%5s' % classes[labels[j]] for j in range(8)))
                print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(8)))
            total += labels.size(0)
            flag = 1
            correct += (predicted==labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return 0


def train():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    iter = 0
    num = 1
    for nEpoch in range(Epochs):
        running_loss = 0
        for i,data in enumerate(trainloader, 0):
            iter = iter+1
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            writer.add_scalar('loss', loss.item(), iter)
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print('epoch: %d\t batch: %d\t loss: %.6f' % (nEpoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                writer.add_scalar('accuracy', Accuracy(), num + 1)
                num = num + 1
        torch.save(net, './model/model.pkl')

if __name__ == '__main__':
    if os.path.exists(modelpath):
        print("Loading existing model!")
        net = torch.load(modelpath)
        print("Model loaded!")
    else:
        print("No existing model!")
    print("-------------Training started------------------")
    #train()
    writer.close()
    Accuracy1()
    print("-------------Training closed--------------------")
