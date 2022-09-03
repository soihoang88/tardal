#----- INIT -----
import torch

from modules.generator import Generator

# init model
DIM, DEPTH = 32, 3
net = Generator(dim=DIM, depth=DEPTH)

# load our pre-trained model
ID = 'weights/tardal.pt' # tardal could be replaced with tardal+ or tardal++
ck_pt = torch.load(ID)
net.load_state_dict(ck_pt)
print(net)


#---- FUSE DEMO -----
from pathlib import Path
from IPython import display
from pipeline.eval import Eval

CUDNN = False # use cudnn boost (recommend only if images are in same size)
HALF = False # use float16 instead of float32 for faster speed
EVAL = '+' in ID # use eval mode for tardal+ and tardal++
COLOR = True # colorize fused image(s) with corresponding visible image(s)

# fuse infrared and visible image(s)
eval = Eval(net, cudnn=CUDNN, half=HALF, eval=EVAL)
path = Path('data/sample/s1')
eval([path / 'ir/M3FD_00471.png'], [path / 'vi/M3FD_00471.png'], Path('runs/sample/s1'), color=COLOR)

# display sample
display.Image('runs/sample/s1/M3FD_00471.png')