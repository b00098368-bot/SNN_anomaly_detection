import torch
import snntorch as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # should print: cuda

lif = snn.Leaky(beta=0.9).to(device)
mem = lif.init_leaky()
x = torch.rand(1, device=device)
spk, mem = lif(x, mem)
print(spk, mem)