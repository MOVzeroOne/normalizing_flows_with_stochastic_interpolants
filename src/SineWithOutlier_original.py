import torch 
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt  
import numpy as np 
from tqdm import tqdm
from torch.utils.data import Dataset

class mlp(nn.Module):
    def __init__(self,struct:list,act_f:nn.Module = nn.ReLU) -> None:
        super().__init__()
        self.struct = struct 
        self.act_f = act_f
        self.layers = self._build()
    
    def _build(self) -> nn.Sequential:
        layers = []
        for index,(i,j) in enumerate(zip(self.struct[:-1],self.struct[1:])):
            layers.append(nn.Linear(i,j))
            if(not (index == len(self.struct)-2)):#if not last layer
                layers.append(self.act_f())
        return nn.Sequential(*layers)

    def forward(self,x:torch.tensor) -> torch.tensor:
        return self.layers(x) 


def I(x0,x1,t):
    """
    t ∈ [0,1]
    """
    return (t)*x1 + (1-t)*x0

def I_grad(x0,x1,t,h=0.000001):
    return (I(x0,x1,t+h)-I(x0,x1,t-h))/(2*h)


class SineWithOutlier(Dataset):
    """
    https://github.com/yuchenmo/MWUGAN_NeurIPS
    
    """
    def __init__(self, dim=2, sig=0.5, num_per_mode=50):
        self.num_per_mode = num_per_mode
        self.sig = sig
        self.isimg = False
    
        uniform_num = 100000
        x = np.random.uniform(-10000, 10000, size=uniform_num)
        y = np.sin(x / 250 * np.pi) * x

        outlier_num = 250
        data2 = np.random.multivariate_normal((0, 10000), cov=np.diag([self.sig for _ in range(2)]).tolist(), size=outlier_num)

        data = np.vstack((x, y)).T
        data = np.vstack((data, data2))
        data /= 1000.

        self.data = torch.FloatTensor(data)
        self.label = torch.LongTensor(np.zeros((uniform_num + outlier_num)))
        self.sample_weights = np.ones(len(self.label))

        self.n_data = len(self.data)
        self.maxval = np.abs(data).max()

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.label)

    def sample_batch_uniform(self, batch_size, to_tensor=False):
        idx = np.random.choice(range(len(self.label)), size=batch_size, replace=True)
        samples = self.data[idx]
        labels = self.label[idx]
        if to_tensor:
            samples = torch.from_numpy(samples)
        return samples, labels

    def sample_batch(self, batch_size, to_tensor=False):
        p = self.sample_weights / np.sum(self.sample_weights)
        idx = np.random.choice(range(len(self.label)), size=batch_size, replace=True, p=p)
        samples = self.data[idx]
        labels = self.label[idx]
        if to_tensor:
            samples = torch.from_numpy(samples)
        return samples, labels

if __name__ == "__main__":
    batch_size = 1000
    sample_size = 1000
    net = mlp([3,100,100,100,2])
    n = 100
    std = 1
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    dataset = SineWithOutlier()


    plt.ion()
    for i in tqdm(range(100000),ascii=True):
        optimizer.zero_grad()
        x1,_ = dataset.sample_batch(batch_size)
        x0 = torch.randn(batch_size,2)*std
        t = torch.rand(batch_size,1)
        x = I(x0,x1,t)
        grad = net(torch.cat((x,t),dim=1))
        loss = torch.sum((torch.abs(grad)**2-2*I_grad(x0,x1,t)*grad))/x0.size(0)

        loss.backward()
        optimizer.step()
        if(i % 100 ==0 and i > 10000):
            with torch.no_grad():
                x0 = torch.randn(sample_size,2)*std
                x1,_ = dataset.sample_batch(sample_size)
                xt = x0
                for t in torch.linspace(0,1,n):

                    xt += net(torch.cat((xt,torch.ones(sample_size,1)*t),dim=1))*(1/n)

                    plt.cla()
                    plt.scatter(x0[:,0],x0[:,1],color="green")
                    plt.scatter(x1[:,0],x1[:,1],color="red")
                    plt.scatter(xt[:,0],xt[:,1],color="blue")
                    plt.pause(0.01)
