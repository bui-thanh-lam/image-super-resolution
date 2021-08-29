from argparse import ArgumentParser
from dataset import SRDataset
from models import *
import torch
from torch.utils.data import DataLoader
from torchvision.io.image import ImageReadMode


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, dataloader, criterion, optimizer):
    """Main function to train a model.

    Args:
        model (PyTorch nn.Module): Bibubic, SRCNN or SRCNNpp.
        iterator (PyTorch Dataloader): a PyTorch dataloader contains a SRDataset.
        criterion (PyTorch nn.Module): MSELoss or CombinedLoss.
        optimizer (PyTorch torch.optim): any PyTorch optimizer.

    Returns:
        loss (float32): loss at current epoch.
    """
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input, target = batch
        input = input.to(DEVICE)
        target = target.to(DEVICE)
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
    return epoch_loss / len(dataloader)


# Add the arguments
args = ArgumentParser()
args.add_argument('--lr_dir', type=str, help='Path to low-res (input) images directory', default="../data/train/LR/")
args.add_argument('--hr_dir', type=str, help='Path to high-res (target) images directory', default="../data/train/HR/")
args.add_argument('--meta_file', type=str, help='Path to the metadata file', default="../data/train/file_names.txt")
args.add_argument('--n_epochs', type=int, help='Number of training epochs', default=20)
args.add_argument('--scale_factor', type=float, help='Scale factor', default=2)
args.add_argument('--model', type=str, help='Name of the model: bicubic, srcnn or srcnnpp', default='srcnnpp')
args.add_argument('--lr', type=float, help='Init learning rate', default=1e-4)
args.add_argument('--loss', type=str, help='Name of the loss function: mse or combined', default='mse')
args.add_argument('--bs', type=int, help='Training batch size', default=64)
args.add_argument('--color_mode', type=str, help='Color mode of training images: gray or rgb', default='rgb')
args.add_argument('--saved_weights_file', type=str, help='Path to the file that stores the weights after training', default="../weights/saved_weights.pt")
args = args.parse_args()

# Hyper params setting and training
if args.color_mode == 'gray':
    N_CHANNELS = 1
    mode = ImageReadMode.GRAY
else:
    N_CHANNELS = 3
    mode = ImageReadMode.RGB

if args.model == 'bicubic': model = Bicubic(scale_factor=args.scale_factor).to(DEVICE)
elif args.model == 'srcnn': model = SRCNN(scale_factor=args.scale_factor).to(DEVICE)
else: model = SRCNNpp(scale_factor=args.scale_factor).to(DEVICE)

if args.loss == 'combined': loss_func = CombinedLoss(pixel_loss_coeff=0.8)
else: loss_func = torch.nn.MSELoss()

optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

train_set = SRDataset(LR_dir=args.lr_dir, HR_dir=args.hr_dir, metadata_file=args.meta_file, mode=mode)

dataloader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=2)


for epoch in range(args.n_epochs):
    print(f"Epoch {epoch+1}:")
    train_loss = train_epoch(model, dataloader, loss_func, optim)
    print(f"\tTrain loss: {train_loss:.4f}")

# Save weights
torch.save(model.state_dict(), args.saved_weights_file)