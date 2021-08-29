import math
from argparse import ArgumentParser
from dataset import SRDataset
from models import *
import torch
from torch.utils.data import DataLoader
from torchvision.io.image import ImageReadMode


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Add the arguments
args = ArgumentParser()
args.add_argument('--lr_dir', type=str, help='Path to low-res (input) images directory', default="../data/test/Set14/LRbicx2/")
args.add_argument('--hr_dir', type=str, help='Path to high-res (target) images directory', default="../data/test/Set14/HR/")
args.add_argument('--meta_file', type=str, help='Path to the metadata file', default="../data/test/Set14/file_names.txt")
args.add_argument('--scale_factor', type=float, help='Scale factor', default=2)
args.add_argument('--model', type=str, help='Name of the model: bicubic, srcnn or srcnnpp', default='srcnnpp')
args.add_argument('--bs', type=int, help='Test batch size', default=1)
args.add_argument('--color_mode', type=str, help='Color mode of test images: gray or rgb', default='rgb')
args.add_argument('--from_pretrained', type=str, help='Path to the file that stores the weights for evaluation', default="../weights/srcnnpp-x2-RGB.pt")
args = args.parse_args()

if args.color_mode == 'gray':
    N_CHANNELS = 1
    mode = ImageReadMode.GRAY
else:
    N_CHANNELS = 3
    mode = ImageReadMode.RGB

if args.model == 'bicubic': model = Bicubic(scale_factor=args.scale_factor).to(DEVICE)
elif args.model == 'srcnn': model = SRCNN(scale_factor=args.scale_factor).to(DEVICE)
else: model = SRCNNpp(scale_factor=args.scale_factor).to(DEVICE)

model.load_state_dict(torch.load(args.from_pretrained))
print(model)

mse = torch.nn.MSELoss()
test_set = SRDataset(LR_dir=args.lr_dir, HR_dir=args.hr_dir, metadata_file=args.meta_file, mode=mode)
dataloader = DataLoader(test_set, batch_size=args.bs, num_workers=2)


model.eval()
with torch.no_grad():
    loss = 0
    for i, batch in enumerate(dataloader):
        input, target = batch
        input = input.to(DEVICE)
        target = target.to(DEVICE)
        output = model(input)
        loss = mse(output, target)
        loss += loss.item()
    loss /= len(dataloader)

print("=======PSNR on test set=======")
print(f"{10 * math.log10(1.0**2 / loss):.2f}")