import os
import torch
from matplotlib import pyplot as plt
from torchvision import transforms, models
from torch import nn
from torchinfo import summary

from imageclassification import data_setup
from imageclassification import engine
from imageclassification import loss_graph
from imageclassification import save__model
from imageclassification.create_model import ImageClassifier


def walk_through_dir(dir_path):
  """"
    Funtion to know what inside the sub_directories or folder.
    Args: 
    input: dir_path --> path where target folder exists.

    output: it will print all the subdirectories and files present inside the folder.
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def device():
  """"
    Funtion to return the device processing the computation of our tensor and other complex computations.
    Args: 
    input: 

    output: it will print all the subdirectories and files present inside the folder.
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print("Device ", device)
  return device

if __name__ == "__main__":
  # change data_1 with your classification data
  image_path = os.getcwd() + "\\data_1"
  walk_through_dir(image_path)

  # train set and test set folder
  train_dir = image_path + "\\training_set\\training_set"
  test_dir = image_path + "\\test_set\\test_set"

  # Transformation of data (Images)
  Height, Width = 224, 224
  IMAGE_SIZE = (Width, Height)
  data_transform = transforms.Compose([
    # Resize the images to IMAGE_SIZE xIMAGE_SIZE
    transforms.Resize(size=IMAGE_SIZE),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
    ])

  # DataLoaders
  NUM_WORKS = os.cpu_count()
  classes_and_loaders = data_setup.create_dataloaders(train_dir=train_dir, test_dir=test_dir, transform=data_transform, batch_size=64, num_workers=NUM_WORKS)
  print(classes_and_loaders)
  train_dataloader, test_dataloader, class_names = classes_and_loaders
  print(f"Train DataLoader: {train_dataloader} | Test DataLoader: {test_dataloader} | Classes names: {class_names}")
  
  ## cpu or gpu device we have?
  device = device()
  print(device)
  
  
  # Model from package
  # MyModel = ImageClassifier().to(device)
  # print(MyModel)

  # VGG16
  MyModel = vgg16 = models.vgg16(pretrained=False)
  MyModel = MyModel.to(device)
  print(MyModel)

  # ============================ Using Pretrained Model ====================
  # MyModel = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  # MyModel = MyModel.to(device)
  # print(MyModel)

  #  train 
  # Set random seeds
  torch.manual_seed(42)
  torch.cuda.manual_seed(42)

  # Setup loss function and optimizer
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params=MyModel.parameters(), lr=1e-3)

  # Start the timer
  from timeit import default_timer as timer
  start_time = timer()
# Increase Number of epochs as your choice
  NUM_EPOCHS = 1

  # Train Your Model
  model_results = engine.train(MyModel,
            train_dataloader,
          test_dataloader,
          optimizer,
          loss_fn,
          NUM_EPOCHS, device=device)
  
  # End the timer and print out how long it took
  end_time = timer()
  print(f"Total training time: {end_time-start_time:.3f} seconds")

  # Summary of model 
  # do a test pass through of an example input size
  model_summary = summary(MyModel, input_size=[1, 3, Width ,Height]) 
  print(f"Model Summary \n{model_summary}")

  loss_graphh = loss_graph.plot_loss_curves(model_results)
  print(loss_graph)

  # Predictions
  # Choose a image.
  custom_image_path = test_dir + "\\dogs\\dog.4001.jpg"

  import torchvision
  # Load in custom image and convert the tensor values to float32
  custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

  # Divide the image pixel values by 255 to get them between [0, 1]
  custom_image = custom_image / 255.

  # Print out image data
  print(f"Custom image tensor:\n{custom_image}\n")
  print(f"Custom image shape: {custom_image.shape}\n")
  print(f"Custom image dtype: {custom_image.dtype}") 

  custom_image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
  ])

  # Transform target image
  custom_image_transformed = custom_image_transform(custom_image)

  # Print out original shape and new shape
  print(f"Original shape: {custom_image.shape}")
  print(f"New shape: {custom_image_transformed.shape}")

  MyModel.eval()
  with torch.inference_mode():
      # Add an extra dimension to image
      custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)

      # Print out different shapes
      print(f"Custom image transformed shape: {custom_image_transformed.shape}")
      print(f"Unsqueezed custom image shape: {custom_image_transformed_with_batch_size.shape}")

      # Make a prediction on image with an extra dimension
      custom_image_pred = MyModel(custom_image_transformed.unsqueeze(dim=0).to(device))
      print(custom_image_pred)
      # Let's convert them from logits -> prediction probabilities -> prediction labels
      # Print out prediction logits
      print(f"Prediction logits: {custom_image_pred}")

      # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
      custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
      print(f"Prediction probabilities: {custom_image_pred_probs}")

      # Convert prediction probabilities -> prediction labels
      custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
      print(f"Prediction label: {custom_image_pred_label}")

      # Checking the what is classified inside the Cat, dog or non of these.
      custom_image_pred_class = class_names[custom_image_pred_label.cpu()] # put pred label to CPU, otherwise will error
      if (custom_image_pred_class == "dogs"):
        print(f"Found: {custom_image_pred_class} 🐶")
      elif(custom_image_pred_class == "cats"):
        print(f"Found: {custom_image_pred_class} 😺")
      else:
        print(f"Found unknown Mujhko pehchanlo mai huu konn")

      # Plot custom image
      plt.imshow(custom_image.permute(1, 2, 0)) # need to permute image dimensions from CWH -> HWC otherwise matplotlib will error
      plt.title(f"Image shape: {custom_image.shape}")
      plt.axis(False)
      plt.xlabel('Label')
      plt.show()

      # Saving Model
      Save_model = save__model.save_model(MyModel)
      # print(Save_model)
 
      try:
        Loadm = torch.load(os.getcwd() + '\\mymodel_script.pt')
        print("Success for 2")
      except:
        print("Failed to Load way 2")
      print(Loadm)