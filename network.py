import torch
from pathlib import Path
import requests
import pickle
import gzip

### DO NOT EDIT THESE LINES ###

# The following lines of code handle downloading the MNIST dataset, and opening
# corresponding file. This is how we obtain the variables x_train, y_train, x_test,
# and y_test.
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/" # link to the mnist dataset
FILENAME = "mnist.pkl.gz" # link to the file that we will be using (digits 0-9)
if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_test y_test), _) = pickle.load(f, encoding="latin-1")

dtype = torch.float # set the data type to float, to preserve decimal places
device = torch.device("cpu") # set device to cpu to optimize runtime

# input_size and output_size are characterstic to the mnist dataset, so do not
# alter them, or else this network will not run correctly (if it all)
input_size = 784 # input size is 28 x 28 = 784, which is charactersitic to the mnist dataset
output_size = 10 # output size is 10 since there are 10 possible digits that the input could be (0-9)

### DO NOT EDIT ABOVE THIS LINE ###

# hyperparameters (these can be changed around)
batch_size = 64
hidden_size = 100
num_epochs = 5
learning_rate = 1e-5

# load in the data obtained from mnist
train_data = torch.from_numpy(x_train)
train_labels = torch.from_numpy(y_train)

# initialize the weight matrices to random numbers, but the sizes must be compatible
# with our network
w1 = torch.randn(input_size, hidden_size, device=device, dtype=dtype)
w2 = torch.randn(hidden_size, output_size, device=device, dtype=dtype)

for epoch in range(num_epochs):
  epoch_loss = 0

  for ind in range(x.shape[0] // batch_size)):
    grad_w1_batch = 0 # track the gradient descent of the w1 matrix
    grad_w2_batch = 0 # track the gradient descent of the w2 matrix
    batch_loss = 0 # track the loss across the batch

    for delta in range(batch_size):
      # in the following code, we will complete a forward pass:
      #     - matrix multiply input with first weight matrix
      #     - apply relu activation
      #     - compute predicted y by matrix multiplying the output of our first
      #       with our second weight matrix. this will then yield a matrix of size
      #       10, where each index contains the probability of the input being
      #       the corresponding number (i.e. index 0 contains P(input == 0))
      #     - calculate loss and apply gradient descent

      # calculate the current batch's starting index (ind * batch_size), and then
      # add delta to get the specific index of our current input
      curr = (ind * batch_size) + delta
      new_x = x[curr].reshape(1,784) # reshape the data from 28x28 to 1x784

      # matrix multiply the input (new_x) with our weight matrix (w1)
      hidden_layer = new_x.mm(w1)

      # we want to use relu activation, so by utilizning clamp() and setting
      # values below 0 -> 0, we can recreate the relu function
      h_relu = hidden_layer.clamp(min=0)

      # multiply to get output
      y_pred = h_relu.mm(w2)

      # compute loss and update tracker variable batch_loss
      loss = (y_pred - y[curr]).pow(2).sum().item()
      batch_loss = (batch_loss + loss) / 2

      # derivative of (y_pred - y[curr])^2 tells us our gradient slope
      grad_y_pred = 2.0 * (y_pred - y[curr])

      # perform gradient descent
      grad_w2 = h_relu.t().mm(grad_y_pred)
      grad_h_relu = grad_y_pred.mm(w2.t())
      grad_h = grad_h_relu.clone() # make a copy of the gradient of the hidden layer
      grad_h[hidden_layer < 0] = 0 # set all values

      # update the gradient tracker variables, so that the weights will be ready
      # to update at the end of the batch sweep
      grad_w1 = new_x.t().mm(grad_h)
      grad_w1_batch += grad_w1
      grad_w2_batch += grad_w2

    epoch_loss += batch_loss

    # backprop to compute gradients of w1 and w2 with respect to loss. note that
    # the weights are only updated as much as the learning_rate allows!
    w1 -= learning_rate * (grad_w1_batch)
    w2 -= learning_rate * (grad_w2_batch)
    if curr % 100 == 99:
      print(curr, epoch_loss/curr)

print("final loss:")
print(epoch_loss/curr)
