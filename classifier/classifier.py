""" Importing """
from logging import raiseExceptions
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random # for torch seed
import os # for torch seed
import re
import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset
import argparse
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
# For results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time
import torchvision
from torch.utils.data import WeightedRandomSampler

# Parse inputs
parser = argparse.ArgumentParser(description='Classifier for defect generation')
parser.add_argument('--dataset', default='mvtec', help='dataset name')
# parser.add_argument('--product', required=True, help='product name')
# parser.add_argument('--defect', required=True, help='defect name')
# parser.add_argument('--epoch', required=True, type=int, help='defect name')
parser.add_argument('--product', default='wood', help='product name')
parser.add_argument('--defect', default='hole', help='defect name')
parser.add_argument('--epoch', default=250, type=int, help='defect name')
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
# parser.add_argument("--debug", action="store_true", help="Enables debugging (a few epochs of training)")
parser.add_argument("--debug", default=False,  help="Enables debugging (a few epochs of training)")
parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--img_size", type=int, default=224, help="Image size")
parser.add_argument("--model", default='resnet18',  help="Model selection: custom, vgg16, resnet18, densenet")
parser.add_argument("--training_type", "--tp", default='scratch',  help="How to train vgg, resnet18 and densenet: from scratch, fine_tuning or features_extraction")

args = parser.parse_args()

DATASET = args.dataset
PRODUCT = args.product
DEFECT = args.defect

# Set paths
normal_samples_paths, defective_samples_paths, fake_samples_paths = list(), list(), list()
dir_path = os.path.dirname(os.path.realpath(__file__))
# OUTPUT_PATH = os.path.join(dir_path, 'output')
# try:
#     os.makedirs(OUTPUT_PATH)
#     print('Output folder created!')
# except OSError:
#     print('WARNING: Output folder already created or problem encountered!')

normal_samples_paths += [os.path.join(os.path.dirname(dir_path), 'data', DATASET, PRODUCT, 'train', 'good')]
normal_samples_paths += [os.path.join(os.path.dirname(dir_path), 'data', DATASET, PRODUCT, 'test', 'good')]
defective_samples_paths += [os.path.join(os.path.dirname(dir_path), 'data', DATASET, PRODUCT, 'test', DEFECT)]
fake_samples_paths += [os.path.join(os.path.dirname(dir_path), 'results', DATASET + '_dataset', 'test', 'standard', 'A2B', 'epoch_' + str(args.epoch))]

normal_files_path, defective_files_path, fake_files_path = list(), list(), list()
for p in normal_samples_paths:
  normal_files_path += [os.path.join(p, f) for f in os.listdir(p)]
random.shuffle(normal_files_path)
for p in defective_samples_paths:
  defective_files_path += [os.path.join(p, f) for f in os.listdir(p)]
random.shuffle(defective_files_path)
for p in fake_samples_paths:
  fake_files_path += [os.path.join(p, f) for f in os.listdir(p) if re.match(r"fake_[0-9]+", f)] # select only defective sample images
random.shuffle(fake_files_path)

# Split data 
# Normal : 60% train - 20% val - 20% test
# Fake defects: 80% train - 20 % val - 0% test
# Real defects: 0% train - 0% val - 100% test
train_files, val_files, test_files = list(), list(), list()
train_labels, val_labels, test_labels = list(), list(), list()
normal_size, defective_size, fake_size = len(normal_files_path), len(defective_files_path), len(fake_files_path)

# weights for cross entropy loss
class_weights = torch.Tensor([1-(normal_size/(normal_size + defective_size)), 1-(defective_size/(normal_size + defective_size))])

train_files += normal_files_path[: int(0.6 * normal_size)]
train_labels += [0] * len(normal_files_path[: int(0.6 * normal_size)])
train_files += fake_files_path[: int(0.8 * fake_size)]
train_labels += [1] * len(fake_files_path[: int(0.8 * fake_size)])
val_files += normal_files_path[int(0.6 * normal_size):int(0.8 * normal_size)]
val_labels += [0] * len(normal_files_path[int(0.6 * normal_size):int(0.8 * normal_size)])
val_files += fake_files_path[int(0.8 * fake_size):]
val_labels += [1] * len(fake_files_path[int(0.8 * fake_size):])
test_files += normal_files_path[int(0.8 * normal_size):]
test_labels += [0] * len(normal_files_path[int(0.8 * normal_size):])
test_files += defective_files_path
test_labels += [1] * len(defective_files_path)

print("-----------------------------------")
print("-------- Dataset statistics -------")
print(f"Normal: 60% train ({len(normal_files_path[: int(0.6 * normal_size)])}) - 20% val ({len(normal_files_path[int(0.6 * normal_size):int(0.8 * normal_size)])}) - 20% test ({len(normal_files_path[int(0.8 * normal_size):])})")
print(f"Fake defects: 80% train ({len(fake_files_path[: int(0.8 * fake_size)])}) - 20 % val ({len(fake_files_path[int(0.8 * fake_size):])}) - 0% test ")
print(f"Real defects: 0% train - 0% val - 100% test ({len(defective_files_path)})")

# Handle class imbalance
class_sample_count = np.array(
    [len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in train_labels])
samples_weight = torch.from_numpy(samples_weight)
samples_weigth = samples_weight.double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

class MyDataset(Dataset):
    def __init__(self, files_list, labels, transform=None):
  
      self.data = files_list
      
      self.labels = labels
      self.transform = transform

    def __getitem__(self, index):
      sample = (Image.open(self.data[index]).convert("RGB"))
      label = self.labels[index]

      if self.transform:
          sample = self.transform(image=np.asarray(sample))
      return sample["image"], label

    def __len__(self):
        return len(self.data)
    
    
# Normalization
# mean_imagenet = [0.485, 0.456, 0.406]
# std_imagenet = [0.229, 0.224, 0.225]
mean = (.5, .5, .5)
std = (.5, .5, .5)

train_transforms = A.Compose([
    A.Resize(int(args.img_size * 1.12), int(args.img_size * 1.12), interpolation=cv2.INTER_CUBIC),
    A.RandomCrop(args.img_size, args.img_size, p=1), 
    A.OneOf([
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.5),
      # A.ShiftScaleRotate (shift_limit=0.15, scale_limit=1.2, rotate_limit=45, interpolation=1, \
      #      border_mode=1, value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None, rotate_method='largest_box', always_apply=False, p=0.5),
    ], p=1),
    # A.ElasticTransform (alpha=1, sigma=2, alpha_affine=0.5, interpolation=cv2.INTER_CUBIC, \
    #     border_mode=cv2.BORDER_REFLECT, value=None, mask_value=0, always_apply=False, approximate=False, \
    #         same_dxdy=False, p=0.5),
    # A.RandomBrightnessContrast(p=0.2),
    # A.RandomContrast(p=0.2),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(args.img_size, args.img_size, interpolation=cv2.INTER_CUBIC),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

# Create data loaders
train_dataset = MyDataset(train_files, train_labels, transform=train_transforms)
val_dataset = MyDataset(val_files, val_labels, transform=val_transforms)
test_dataset = MyDataset(test_files, test_labels, transform=val_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output

OUTPUT_DIM = 2

if args.training_type in ['fine_tuning', 'feature_extraction']:
    pretrained = True
    print('Fine_tuning or feature_extraction activated.')
else:
    pretrained = False

if args.model == 'custom':
    model=Net()
elif args.model == 'vgg16':
    model = torchvision.models.vgg16(pretrained = pretrained)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, OUTPUT_DIM)
    if args.training_type == 'features_extraction':
        # Freeze all "features" layers
        for parameter in model.features.parameters():
          parameter.requires_grad = False


elif args.model == 'resnet18':
    model = torchvision.models.resnet18(pretrained=pretrained)
    if args.training_type == 'features_extraction':
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, OUTPUT_DIM)
        
          
elif args.model == 'densenet':
    model = torchvision.models.densenet161(pretrained=pretrained)
    if args.training_type == 'features_extraction':
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    model.classifier = nn.Linear(model.classifier.in_features, OUTPUT_DIM)    
else:
    raise ValueError('Model not recognized. Please, verify input model name.')


# Loss
criterion = nn.CrossEntropyLoss() # Softmax + CrossEntropy

# Put model&criterion on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = criterion.to(device)

lr = 2e-4

# Optim
optimizer = optim.Adam(model.parameters(), lr=lr)

model = model.to(device)

"""* Training phase"""

def train(model, iterator, optimizer, criterion, device):
  epoch_loss = 0
  epoch_acc = 0

  # Train mode
  model.train()

  for (x,y) in iterator:
    x = x.to(device)
    y = y.to(device)
    # Set gradients to zero
    optimizer.zero_grad()
    
    with torch.set_grad_enabled(True):
      # Make Predictions
      y_pred = model(x)
      # Compute loss
      loss = criterion(y_pred, y)
    
    # Compute accuracy
    acc = calculate_accuracy(y_pred, y)

    # Backprop
    loss.backward()

    # Apply optimizer
    optimizer.step()

    # Extract data from loss and accuracy
    epoch_loss += loss.item()
    epoch_acc += acc.item()

  return epoch_loss/len(iterator), epoch_acc/len(iterator)

"""* Validation/Testing phase"""

def evaluate(model, iterator, criterion, device):
  epoch_loss = 0
  epoch_acc = 0

  # Evaluation mode
  model.eval()

  # Do not compute gradients
  with torch.no_grad():

    for(x,y) in iterator:

      x = x.to(device)
      y = y.to(device)
      
      # Make Predictions
      y_pred = model(x)

      # Compute loss
      loss = criterion(y_pred, y)
      
      # Compute accuracy
      acc = calculate_accuracy(y_pred, y)

      # Extract data from loss and accuracy
      epoch_loss += loss.item()
      epoch_acc += acc.item()

  return epoch_loss/len(iterator), epoch_acc/len(iterator)

def calculate_accuracy(y_pred, y):
  '''
  Compute accuracy from ground-truth and predicted labels.
  
  Input
  ------
  y_pred: torch.Tensor [BATCH_SIZE, N_LABELS]
  y: torch.Tensor [BATCH_SIZE]

  Output
  ------
  acc: float
    Accuracy
  '''
  y_prob = F.softmax(y_pred, dim = -1)
  y_pred = y_pred.argmax(dim=1, keepdim = True)
  correct = y_pred.eq(y.view_as(y_pred)).sum()
  acc = correct.float()/y.shape[0]
  return acc

def model_training(n_epochs, model, train_iterator, valid_iterator, optimizer, criterion, device, model_name='best_model.pt'):

  # Initialize validation loss
  best_valid_loss = float('inf')

  # Save output losses, accs
  train_losses = []
  train_accs = []
  valid_losses = []
  valid_accs = []

  # Loop over epochs
  for epoch in range(n_epochs):
    start_time = time.time()
    # Train
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    # Validation
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
    # Save best model
    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      # Save model
      torch.save(model.state_dict(), model_name)
    end_time = time.time()
    
    print(f"\nEpoch: {epoch+1}/{n_epochs} -- Epoch Time: {end_time-start_time:.2f} s")
    print("---------------------------------")
    print(f"Train -- Loss: {train_loss:.3f}, Acc: {train_acc * 100:.2f}%")
    print(f"Val -- Loss: {valid_loss:.3f}, Acc: {valid_acc * 100:.2f}%")

    # Save
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    valid_losses.append(valid_loss)
    valid_accs.append(valid_acc)

  return train_losses, train_accs, valid_losses, valid_accs

train_losses, train_accs, valid_losses, valid_accs = model_training(args.epochs, 
                                                                    model, 
                                                                    train_loader, 
                                                                    val_loader, 
                                                                    optimizer, 
                                                                    criterion, 
                                                                    device,
                                                                    'net.pt')

def model_testing(model, test_iterator, criterion, device, model_name='best_model.pt'):
  # Test model
  model.load_state_dict(torch.load(model_name))
  test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
  print(f"Test -- Loss: {test_loss:.3f}, Acc: {test_acc * 100:.2f} %")

model_testing(model, test_loader, criterion, device, 'net.pt')

def predict(model, iterator, device):
  
  # Evaluation mode
  model.eval()
  
  labels = []
  pred = []

  with torch.no_grad():
    for (x, y) in iterator:
      x = x.to(device)
      y_pred = model(x)

      # Get label with highest score
      y_prob = F.softmax(y_pred, dim = -1)
      top_pred = y_prob.argmax(1, keepdim=True)

      labels.append(y.cpu())
      pred.append(top_pred.cpu())

  labels = torch.cat(labels, dim=0)
  pred = torch.cat(pred, dim=0)
  
  return labels, pred

def print_report(model, test_iterator, device):
  labels, pred = predict(model, test_iterator, device)
  print(confusion_matrix(labels, pred))
  print("\n")
  print(classification_report(labels, pred))

print_report(model, test_loader, device)


