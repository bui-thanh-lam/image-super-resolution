from argparse import ArgumentParser
from models import *
import torch
from torchvision.io.image import ImageReadMode
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Add the arguments
args = ArgumentParser()
args.add_argument('--lr_file', type=str, help='Path to a low-res (input) image file', default="../data/test/LR_sample_image.png")
args.add_argument('--sr_file', type=str, help='Path to store the super-res (output) image', default="../data/test/SR_sample_image.png")
args.add_argument('--scale_factor', type=float, help='Scale factor', default=2)
args.add_argument('--model', type=str, help='Name of the model: bicubic, srcnn or srcnnpp', default='srcnnpp')
args.add_argument('--color_mode', type=str, help='Color mode of test images: gray or rgb', default='rgb')
args.add_argument('--from_pretrained', type=str, help='Path to the file that stores the weights for prediction', default="../weights/srcnnpp-x2-RGB.pt")
args = args.parse_args()

if args.color_mode == 'gray':
    N_CHANNELS = 1
    mode = ImageReadMode.GRAY
    pil_mode = 'L'
else:
    N_CHANNELS = 3
    mode = ImageReadMode.RGB
    pil_mode = 'RGB'

if args.model == 'bicubic': 
    model = Bicubic(scale_factor=args.scale_factor).to(DEVICE)
elif args.model == 'srcnn': 
    model = SRCNN(scale_factor=args.scale_factor).to(DEVICE)
    model.load_state_dict(torch.load(args.from_pretrained))
else: 
    model = SRCNNpp(scale_factor=args.scale_factor).to(DEVICE)
    model.load_state_dict(torch.load(args.from_pretrained))

model.load_state_dict(torch.load(args.from_pretrained))
print(model)

lr_img = torch.mul(1.0 / 255, read_image(args.lr_file, mode).float()).to(DEVICE)

model.eval()
with torch.no_grad():
    output = model(lr_img.unsqueeze(0))
sr_img = torch.mul(255.0, output).squeeze(0)
sr_img = sr_img.clamp(min=0, max=255)
sr_img = to_pil_image(sr_img, pil_mode)
sr_img.save(args.sr_file)