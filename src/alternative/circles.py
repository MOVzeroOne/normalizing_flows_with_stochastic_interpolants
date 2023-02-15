import torch 
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt  
import numpy as np 
from tqdm import tqdm 

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


def circles(num_samples,radius=1,std_x=0,std_y=0):
    theta = torch.rand(num_samples,1)*2*torch.pi
    x = (torch.ones(num_samples,1)*-1 + radius*torch.cos(theta))+torch.randn(num_samples,1)*std_x
    y = (torch.ones(num_samples,1)*-1 + radius*torch.sin(theta))+torch.randn(num_samples,1)*std_y
    circle_1 = torch.cat((x,y),dim=1)

    x = (torch.ones(num_samples,1)*2 + radius*torch.cos(theta))+torch.randn(num_samples,1)*std_x
    y = (torch.ones(num_samples,1)*2 + radius*torch.sin(theta))+torch.randn(num_samples,1)*std_y
    circle_2 = torch.cat((x,y),dim=1) 

    circle_1_2 = torch.cat((circle_1,circle_2),dim=0)

    chosen_indices = np.random.choice(torch.arange(circle_1_2.size(0)),num_samples)
    return circle_1_2[chosen_indices]



if __name__ == "__main__":
    batch_size = 100
    sample_size = 200
    net = mlp([3,100,100,100,2])
    n = 100
    optimizer = optim.Adam(net.parameters(),lr=0.001)

    plt.ion()
    for i in tqdm(range(100000),ascii=True):
        optimizer.zero_grad()
        x1 = circles(batch_size)
        x0 = torch.randn(batch_size,2)*2
        t = torch.rand(batch_size,1)
        x = I(x0,x1,t)
        grad = net(torch.cat((x,t),dim=1))
        loss = nn.MSELoss()(grad,I_grad(x0,x1,t))

        loss.backward()
        optimizer.step()
        if(i % 1000 ==0):
            with torch.no_grad():
                x0 = torch.randn(sample_size,2)*2
                x1 = circles(sample_size)
                xt = x0
                for t in torch.linspace(0,1,n):

                    xt += net(torch.cat((xt,torch.ones(sample_size,1)*t),dim=1))*(1/n)

                    plt.cla()
                    plt.scatter(x0[:,0],x0[:,1],color="green")
                    plt.scatter(x1[:,0],x1[:,1],color="red")
                    plt.scatter(xt[:,0],xt[:,1],color="blue")
                    plt.pause(0.01)
