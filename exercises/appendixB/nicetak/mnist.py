import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


class Mnist:
    def __init__(self, n_batch=64, width=64, depth=2, is_train_small=False):

        self.n_batch = n_batch
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        self.data_train = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)

        if is_train_small:
            self.data_train = Subset(
                self.data_train, range(len(self.data_train) // 10))

        self.loader_train = DataLoader(
            self.data_train, batch_size=self.n_batch)

        self.data_test = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        self.loader_test = DataLoader(self.data_test, batch_size=self.n_batch)

        self.net = Net(width=width, depth=depth)

    def train(self, optimizer="ADAM", lr=0.001, n_epochs=20, verbose=False):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.n_epochs = n_epochs

        if optimizer == "ADAM":
            opt = optim.Adam(self.net.parameters(), lr=lr)
        elif optimizer == "SGD":
            opt = optim.SGD(self.net.parameters(), lr=lr)
        else:
            return print("Unsupported Optimizer")

        self.history = {
            "loss_train": [],
            "acc_test": []
        }

        for epoch in range(n_epochs):
            # Train
            for x, y in self.loader_train:
                x, y = x.to(self.device), y.to(self.device)

                self.net.zero_grad()
                output = self.net(x.view(-1, 28 * 28))
                loss = f.nll_loss(output, y)
                loss_train = loss.item()
                loss.backward()
                opt.step()

            self.history["loss_train"].append(loss_train)

            # Test
            acc = 0.0
            with torch.no_grad():
                for x, y in self.loader_test:
                    x, y = x.to(self.device), y.to(self.device)

                    output = self.net(x.view(-1, 28 * 28))
                    pred = output.argmax(dim=1, keepdim=True)
                    acc += pred.eq(y.view_as(pred)).sum().item()

            acc /= len(self.loader_test.dataset)

            self.history["acc_test"].append(acc)

            if verbose:
                print(
                    f"Epoch {epoch}: loss {loss_train:.{3}f} & accuracy {acc:.{3}f}")


class Net(nn.Module):
    def __init__(self, width, depth, dim_input=28 * 28, dim_output=10):
        super(Net, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(dim_input, width))
        for _ in range(depth):
            self.layers.append(nn.Linear(width, width))
        self.layers.append(nn.Linear(width, dim_output))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = f.relu(layer(x))

        out = f.log_softmax(self.layers[-1](x), dim=1)

        return out
