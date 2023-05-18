import torch 
import torch.nn as nn 
from torch import vmap 
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

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.layers(x) 
    
def I(x0,x1,t):
    """
    t ∈ [0,1]
    """
    return (t)*x1 + (1-t)*x0

def I_grad(x0,x1,t,h=0.000001):
    return (I(x0,x1,t+h)-I(x0,x1,t-h))/(2*h)

if __name__ == "__main__":
    #hyperparameters 
    lr = 0.0001
    n = 100
    iterations = 10000
    batch_size = 128

    #structures
    x0 = torch.tensor([[0.0,1.0],[0.0,-1.0]]) 
    x1 = torch.tensor([[1.0,-1.0],[1.0,1.0]])
    t = torch.linspace(0,1,n)
    xt = vmap(I,(None,None,0))(x0,x1,t)
    t = t.view(-1,1,1).repeat((1,2,1))
    xt_t = torch.cat((xt,t),dim=2)
    
    network = mlp([3,100,100,2])
    optimizer = optim.Adam(network.parameters(),lr=lr)

    plt.ion()
    for i in range(iterations):
        indices = np.random.choice(np.arange(n),size=batch_size,replace=True)
        selected_t = t[indices]
        selected_xt_t = xt_t[indices]
        grad_xt = vmap(I_grad,(None,None,0))(x0,x1,selected_t)
        optimizer.zero_grad()
        loss = nn.MSELoss()(network(selected_xt_t),grad_xt)
        loss.backward()
        optimizer.step()

        if(i %1000 == 0):
            with torch.no_grad():
                xt = torch.clone(x0)
                for index in tqdm(range(n),ascii=True):
                    #euler solver 
                    current_t = t[index]
                    xt += network(torch.cat((xt,current_t),dim=1))*(1/n)

                    plt.cla()
                    plt.scatter(x0[:,0],x0[:,1],color="green")
                    plt.scatter(x1[:,0],x1[:,1],color="red")
                    plt.scatter(xt[:,0],xt[:,1],color="blue")
                    plt.pause(0.1)
    