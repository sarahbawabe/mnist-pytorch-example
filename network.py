import torch
from pathlib import Path
import requests
import pickle
import gzip

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

dtype = torch.float

device = torch.device("cpu")

# N - batch size
# D_in - input dim
# H - hidden dim
# D_out - output dim
N, D_in, H, D_out = 64, 784, 100, 10

# random data
#x = torch.randn(N,D_in,device=device,dtype=dtype)
x = torch.from_numpy(x_train)
#y = torch.randn(N,D_out, device=device,dtype=dtype)
y = torch.from_numpy(y_train)

# random weights
w1 = torch.randn(D_in,H,device=device,dtype=dtype)
w2 = torch.randn(H,D_out,device=device,dtype=dtype)

learning_rate = 1e-5
for epoch in range(5): # 5 epochs
  epoch_loss = 0
  for ind in range(round(50000/N)): # size of each tensor
    grad_w1_batch = 0
    grad_w2_batch = 0
    batch_loss = 0
    for delta in range(N): # batch size
      # Forward pass: compute predicted y
      ## .mm -> matrix multiplication
      curr = (ind * N) + delta
      new_x = x[curr].reshape(1,784)
      h = new_x.mm(w1)
      ## use relu activation, clamp is setting values below 0 -> 0 here
      h_relu = h.clamp(min=0)
      ## Now multiply to get output
      y_pred = h_relu.mm(w2)

      # Compute and print loss
      loss = (y_pred - y[curr]).pow(2).sum().item()
      batch_loss = (batch_loss + loss)/2

      grad_y_pred = 2.0 * (y_pred - y[curr])
      # what we want
      grad_w2 = h_relu.t().mm(grad_y_pred)
      grad_h_relu = grad_y_pred.mm(w2.t())
      grad_h = grad_h_relu.clone()
      grad_h[h < 0] = 0
      # what we want
      grad_w1 = new_x.t().mm(grad_h)
      grad_w1_batch += grad_w1
      grad_w2_batch += grad_w2

    epoch_loss += batch_loss
    # Backprop to compute gradients of w1 and w2 with respect to loss
    # print(grad_w1_all/N)
    # print(grad_w2_all/N)
    w1 -= learning_rate * (grad_w1_batch)
    w2 -= learning_rate * (grad_w2_batch)
    if curr % 100 == 99:
      print(curr, epoch_loss/curr)

print("final loss:")
print(epoch_loss/curr)
