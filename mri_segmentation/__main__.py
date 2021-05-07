import torch
import torch.optim as optim

from .utils import set_seed, train
from .dataset import load_MRIDataset
from .model import UNet

# Load data
set_seed(1234)
(trainloader, validloader), (train_dataset, valid_dataset) = load_MRIDataset('data/', batch_size=32, patients_valid=5)

# Train model
num_epochs = 85
model = UNet()
optimizer = optim.Adam(model.parameters())
train(model, trainloader, validloader, optimizer, num_epochs=num_epochs)
torch.save(model.state_dict(), 'weights/model_UNet_num_epochs_{num_epochs}_seed_{seed}.pt')
