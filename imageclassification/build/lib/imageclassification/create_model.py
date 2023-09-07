
import torch
from torch import nn
"""
Contains PyTorch model code for image classification.
"""
# Creating a CNN-based image classifier.
class ImageClassifier(nn.Module):
  """
  Creates Model
  Args:
  input_shape: An Integer indicating number of input channels.
  hidden_units: An Integer indicating number of hidden units between layers.
  output_shapes: An Integer indicating number of outputs units.

  Returns:
  forward pass
  """
  def __init__(self):
    super().__init__()
    self.conv_layer_1 = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(2))
    self.conv_layer_2 = nn.Sequential(
    nn.Conv2d(64, 512, 3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(512),
    nn.MaxPool2d(2))
    self.conv_layer_3 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(512),
    nn.MaxPool2d(2))
    self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=512*3*3, out_features=2))

  def forward(self, x: torch.Tensor):
    x = self.conv_layer_1(x)
    x = self.conv_layer_2(x)
    x = self.conv_layer_3(x)
    x = self.conv_layer_3(x)
    x = self.conv_layer_3(x)
    x = self.conv_layer_3(x)
    x = self.classifier(x)
    return x

# Instantiate an object.
# model = ImageClassifier().to(device)
# model