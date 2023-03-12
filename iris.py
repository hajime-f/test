import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
xtrain, xtest, ytrain, ytest = train_test_split(
    iris.data, iris.target, test_size=0.5)

xtrain = torch.from_numpy(xtrain).type('torch.FloatTensor')
ytrain = torch.from_numpy(ytrain).type('torch.LongTensor')
xtest = torch.from_numpy(xtest).type('torch.FloatTensor')
ytest = torch.from_numpy(ytest).type('torch.LongTensor')


class TwoLayerNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(TwoLayerNN, self).__init__()
        self.l1 = nn.Linear(n_input, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        h1 = torch.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2


model = TwoLayerNN(4, 6, 3)
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_function = nn.CrossEntropyLoss()

model.train()
for i in range(1000):
    output = model(xtrain)
    loss = loss_function(output, ytrain)

    breakpoint()

    print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    output1 = model(xtest)
    ans = torch.argmax(output1, 1)
    print(((ytest == ans).sum().float() / len(ans)).item())
