import torch
from torch.utils.data import DataLoader, Dataset
import numpy
from model import MLP

dev = 'cuda'

class DDFDataset(Dataset):
    def __init__(self,):
        data_load = numpy.load('./bunny.npz')
        self.data = torch.tensor(data_load['arr_0'], device=dev)
        self.points, self.dir, self.val = torch.hsplit(self.data, [3,5])

    def __len__(self,):
        return self.data.size()[0]

    def __getitem__(self, idx):
        return self.points[idx], self.dir[idx], self.val[idx]

dataset = DDFDataset()
ddfload = DataLoader(dataset, batch_size=1024, shuffle=True)


class ClampLoss(torch.nn.Module):
    def __init__(self, eps = .1):
        super().__init__()
        self.eps = eps

    def forward(self, true, pred):
        return torch.sum(torch.abs(
        torch.clamp(pred, min=-self.eps, max=self.eps)-
        torch.clamp(true, min=-self.eps, max=self.eps)))

ddf_model = MLP().to(dev)
optim = torch.optim.Adam(params=ddf_model.parameters(), lr=1e-7)
loss_fn = ClampLoss()
# loss_fn = torch.nn.MSELoss()

data_len = len(ddfload)
epochs = 100

loss_data = list()

print("Model: ", ddf_model)
print("Loss function: ", loss_fn)
print("Optimizer: ", optim)

def plot():
    from matplotlib import pyplot
    pyplot.style.use('bmh')

    pyplot.plot(loss_data)
    pyplot.xlabel('No. of iteration')
    pyplot.ylabel('Loss')
    # pyplot.title('Loss v. Iteration for DDF')
    pyplot.savefig('loss.png')

try:
    for ep in range(epochs):
        count = 0
        for point, dir, val in ddfload:
            pred = ddf_model(point, dir)
            loss = loss_fn(pred, val)

            ddf_model.zero_grad()
            loss.backward()
            optim.step()

            print("[%6.2lf%%] Epoch %3d: Loss = %3.06lf"%
                ((1+count)*100./data_len, ep, loss),
                flush=True,end='\r')
            count += 1

        loss_data.append(loss.item())
        torch.save(ddf_model.state_dict(), './weights/%08d.pt'%ep)
        torch.save(ddf_model.state_dict(), './weights/best_model.pt')
        print(end='\n')
        plot()

except KeyboardInterrupt:
    plot()
