import torch
import torchvision
from torch.utils.data import random_split
from torchvision import transforms
from torch import nn, optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from PIL import Image



batch = 64
T = torchvision.transforms.Compose(
    [torchvision.transforms.Grayscale(num_output_channels=1), torchvision.transforms.ToTensor()])
root = "./MNIST_JPGS/trainingSet/trainingSet"
data = ImageFolder(root, transform=T)
img = Image.open("MNIST_JPGS/trainingSet/trainingSet/7/img_6.jpg",)
img_tensor = transforms.ToTensor()(img).unsqueeze_(0)

train_size = int(0.8*len(data))
val_size = len(data) - train_size
train_data, val_data = random_split(data, [train_size, val_size])

# val_data = ImageFolder('./MNSIT/trainingSet/trainingSet/', transform=T)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch, shuffle=True)

# for d in train_data_loader:
#     print(d)
#
#     break

#x, y = d[0][0], data[1][0]
#print(y.shape)

# plt.imshow(d[0][0].view(28, 28))
# plt.show()

total = 0
counter_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

for da in train_data_loader:
    Xs, ys = da
    for y in ys:
        counter_dict[int(y)] += 1
        total += 1

#print(counter_dict)

# for i in counter_dict:
#     print(f"{i}: {counter_dict[i] / total * 100}")


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # define Layers
        self.fc1 = nn.Linear((28 * 28), 64)  # input and output Linear:fully connected
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):  # feed-forward net
        x = F.relu(self.fc1(x))  # Rectified Linear Activation Function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # multiclass variance
        return F.log_softmax(x, dim=1)


net = Network()
#print(net)

X = torch.rand((28, 28))
X = X.view(-1, 28 * 28)  # -1 is to show that you are not aware of the shape (any size data input)
out = net(X)
#print(out)

# Transfer Learning
optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 5
for epoch in range(EPOCHS):
    for single_data in train_data_loader:
        X, y = single_data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        # loss for single data value is calculated with nll_loss
        loss = F.nll_loss(output, y)
        loss.backward()
        #adjust weights
        optimizer.step()

    #print(loss)

correct = 0
total = 0

with torch.no_grad():
    for d in val_data_loader:
        X, y = d
        #print(X[0])
        #print(y[0])

        output = net(X.view(-1, 28*28))
        #print(output)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]: # this is the output guessed by Neural Network
                correct += 1
            total += 1

print(torch.argmax(net((img_tensor).view(-1, 784))))
print("Accuracy: ", round(correct/total, 3))

