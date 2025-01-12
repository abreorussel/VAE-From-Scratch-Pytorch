import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as  np
import torchvision
from einops import rearrange

from dataset.mnist_loader import MnistDataset
from model.vae import VAEModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the data set and the data loader
mnist = MnistDataset('train', im_path='data/train')
mnist_test = MnistDataset('test', im_path='data/test')
mnist_loader = DataLoader(mnist, batch_size=64, shuffle=True, num_workers=0)

# Instantiate the model
model = VAEModel().to(device)

# Specify training parameters
num_epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)
criterion = torch.nn.MSELoss()

recon_losses = []
kl_losses = []
losses = []
# Run training for 10 epochs
for epoch_idx in range(num_epochs):
    for im, label in tqdm(mnist_loader):
        im = im.float().to(device)
        optimizer.zero_grad()
        mean, log_var, out = model(im)
        
        cv2.imwrite('input.jpeg', 255*((im+1)/2).detach().cpu().numpy()[0, 0])
        cv2.imwrite('output.jpeg', 255 * ((out + 1) / 2).detach().cpu().numpy()[0, 0])
        
        kl_loss = torch.mean(0.5* torch.sum(torch.exp(log_var) + mean**2 - 1 -log_var, dim=-1))
        recon_loss = criterion(out, im)
        loss = recon_loss + 0.00001 * kl_loss
        recon_losses.append(recon_loss.item())
        losses.append(loss.item())
        kl_losses.append(kl_loss.item())
        loss.backward()
        optimizer.step()
    print('Finished epoch:{} | Recon Loss : {:.4f} | KL Loss : {:4f}'.format(
        epoch_idx+1,
        np.mean(recon_losses),
        np.mean(kl_losses)
    ))
    
print('Done Training ...')
# Run a reconstruction for some sample test images
idxs = torch.randint(0, len(mnist_test)-1, (100, ))
ims = torch.cat([mnist_test[idx][0][None, :] for idx in idxs]).float()

_, _, generated_im = model(ims)

ims = (ims + 1)/ 2
generated_im = 1- (generated_im + 1) / 2
out = torch.hstack([ims, generated_im])
output = rearrange(out, 'b c h w -> b () h (c w)')
grid = torchvision.utils.make_grid(output, nrow=10)
img = torchvision.transforms.ToPILImage()(grid)
img.save('reconstruction.png')
print('Done Reconstruction...')