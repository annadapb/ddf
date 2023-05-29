import torch
from torch import nn
# from torch import func
from process_mesh import sample_hemisphere


class MLP(nn.Module):
    # def __init__(self,):
    #     super().__init__()
    #     layer_width = [5, 32, 64, 128, 256, 256, 256, 128, 64, 32, 16, 1]
    #     # layer_width = [5, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 1]
    #     self.layers = nn.ModuleList()
    #     for i in range(1, len(layer_width)):
    #         self.layers.append(nn.Linear(layer_width[i-1], layer_width[i]))
    #         self.layers.append(nn.BatchNorm1d(layer_width[i]))
    #         self.layers.append(nn.ReLU())
    #         self.layers.append(nn.Dropout(p=.5))
    #     self.layers.append(nn.Tanh())

    def __init__(self,):
        super().__init__()
        layer_width = [5, 512, 512, 512, 512, 512, 512, 512, 1]
        # layer_width = [5, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 1]
        self.layers = nn.ModuleList()
        for i in range(1, len(layer_width)):
            self.layers.append( nn.utils.weight_norm(
                nn.Linear(layer_width[i-1], layer_width[i]) ))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=.2))
        self.layers.append(nn.Tanh())


    def trace(self, pos, dir):
        ''' pos: tensor.Size([3])
            dir: tensor.Size([n, 3])'''
        pos = pos.squeeze()
        x = torch.stack([torch.hstack([
            pos,
            # Cartesian to polar conversion next
            torch.atan(dir[i, 1]/dir[i, 0]),
            torch.acos(dir[i, 2])
            ]) for i in range(len(dir))])
        for L in self.layers:
            x = L(x)
        return x

    def forward(self, pos, dir):
        # x = func.vmap(torch.hstack, in_dims=(None, 0))(pos, dir)
        # x = torch.stack([torch.hstack([
        #     pos[i],
        #     # Cartesian to polar conversion next
        #     torch.atan(dir[i, 1]/dir[i, 0]),
        #     torch.acos(dir[i, 2])
        #     ]) for i in range(len(dir))])
        x = torch.hstack((pos, dir))
        for L in self.layers:
            x = L(x)
        return x

    # def grad(self, pos):
    #     dir = sample_hemisphere(pos.shape[0])
    #     x = torch.hstack((pos, dir))
    #     for L in self.layers:
    #         x = L(x)
    #     print(pos.grad)
    #     exit()
    #     return x.grad




        








if __name__=='__main__':
    x = torch.randn(10, 5)
    model = MLP()
    pos = torch.randn(3);
    dir = torch.randn(100, 3);
    z = model.trace(pos, dir)
    print(z)

