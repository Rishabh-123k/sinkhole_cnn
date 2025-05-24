import numpy as np
import torch
from model import SinkholeCNN

# load model
model = SinkholeCNN(input_length=128)
model.load_state_dict(torch.load("../models/sinkhole_cnn.pth"))
model.eval()

# pick a profile to test:
data = np.load("../data/synthetic.npz")
idx = np.random.randint(len(data['sinkholes']))
prof_sink = data['sinkholes'][idx]
prof_flat  = data["non_sink"][idx]

# inference helper
def infer(profile):
    with torch.no_grad():
        p = model(torch.from_numpy(profile).unsqueeze(0)).item()
    return p

print("Sinkhole sample :", infer(prof_sink))
print("Non-sink sample :", infer(prof_flat))
