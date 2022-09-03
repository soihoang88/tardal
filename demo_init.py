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