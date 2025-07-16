import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

from PIL import Image
import pandas as pd
import numpy as np
import os
import wandb
import random

# # Set your WandB API key
# wandb_api_key = "9c50c143446d6356e47417e56da457211e7c3029"

# # Initialize WandB with your API key
# wandb.login(key=wandb_api_key)

# # Initialize wandb with your project name
# wandb.init(project="GAN")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
batch_size = 32

class CustomDataset(Dataset):
    def __init__(self, img_folder, csv_file, transform=None):
        self.img_folder = img_folder
        self.csv_file = csv_file
        self.transform = transform

        # Read CSV file containing image filenames and one-hot encoded labels
        self.data_info = pd.read_csv(csv_file)

        # Number of items in the dataset
        self.data_len = len(self.data_info)

    def __getitem__(self, index):
        # Get image name from the dataframe
        img_name = os.path.join(self.img_folder, self.data_info.iloc[index, 0] + '.jpg') # Add file extension

        # Open image
        image = Image.open(img_name)

        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)

        # # Get one-hot encoded label from the dataframe
        label = self.data_info.iloc[index, 1:].values.astype(np.float32) # Convert labels to float32

        # # Convert one-hot encoded label to normal label
        label = np.argmax(label)
        return image, label

    def __len__(self):
        return self.data_len

transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5 for _ in range(3)], [0.5 for _ in range(3)]),])


# Example dataset instantiation
dataset = CustomDataset(img_folder='/csehome/m23mac008/dl4/Train_data',
                        csv_file='/csehome/m23mac008/dl4/Assignment_4/Train/Train_labels.csv',
                        transform=transform)

# Example dataloader creation
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)

batch_demo = next(iter(trainloader))
print('a',batch_demo[0].shape)
print('b',batch_demo[1].shape)


def load_random_image(folder_path, target_size=(10, 10)):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    random_image_filename = random.choice(image_files)
    random_image_path = os.path.join(folder_path, random_image_filename)

    image = Image.open(random_image_path)
   
    # Apply transformations
    transform_rand = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    image_tensor = transform_rand(image)

    return image, image_tensor

# Example usage:
folder_path = "/csehome/m23mac008/dl4/Assignment_4/Train/Contours"
random_image, random_image_tensor = load_random_image(folder_path)

print('c',random_image_tensor.shape)

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        
        self.conv1 = nn.Conv2d(channels_img + 1, features_d, kernel_size=4, stride=2, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(features_d * 2, affine=True)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm3 = nn.InstanceNorm2d(features_d * 4, affine=True)
        self.leaky_relu3 = nn.LeakyReLU(0.2)
        self.conv4 = nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm4 = nn.InstanceNorm2d(features_d * 8, affine=True)
        self.leaky_relu4 = nn.LeakyReLU(0.2)
        self.conv5 = nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0)
        
        self.embed = nn.Embedding(num_classes, image_size * image_size)

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.image_size, self.image_size)
        x = torch.cat([x, embedding], dim=1)
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.norm2(self.conv2(x)))
        x = self.leaky_relu3(self.norm3(self.conv3(x)))
        x = self.leaky_relu4(self.norm4(self.conv4(x)))
        x = self.conv5(x)
        return x

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, image_size, embed_size):
        super(Generator, self).__init__()
        self.embed = nn.Embedding(num_classes, embed_size)
        self.image_size = image_size
        
        self.conv1 = nn.ConvTranspose2d(channels_noise + embed_size, features_g * 16, kernel_size=4, stride=1, padding=0, bias=False)
        self.norm1 = nn.BatchNorm2d(features_g * 16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(features_g * 8)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(features_g * 4)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm4 = nn.BatchNorm2d(features_g * 2)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm5 = nn.BatchNorm2d(features_g)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.ConvTranspose2d(features_g, channels_img, kernel_size=4, stride=2, padding=1)
        self.upsample = nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=True)
        self.tanh = nn.Tanh()

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.relu5(self.norm5(self.conv5(x)))
        x = self.tanh(self.upsample(self.conv6(x)))
        return x

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


## Check the models with random noise and random labels
image_size = 64
# labels 
num_classes = 7
labels = torch.randint(0, num_classes, (16,)).to(device)
gen = Generator(channels_noise=100, channels_img=3, 
                features_g=64, num_classes=num_classes, image_size=image_size, embed_size=100).to(device)
critic = Discriminator(channels_img=3, features_d=64,
                       num_classes=num_classes, image_size=image_size).to(device)
initialize_weights(gen)
initialize_weights(critic)
x = torch.randn((16, 100, 1, 1)).to(device)
gen_out = gen(x, labels)
print(gen_out.shape)
disc_out = critic(gen_out, labels)
print(disc_out.shape)






def gradient_penalty(critic, real, labels, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images,labels)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty



#Hyperparameters etc.
LEARNING_RATE = 1e-4
CRITIC_ITERATIONS = 5
LAMBDA_GP = 100
NUM_EPOCHS = 100

# # initialize gen and disc, note: discriminator should be called critic,
# # according to WGAN paper (since it no longer outputs between [0, 1])


# # initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# for tensorboard plotting
step = 0

gen.train()
critic.train()


for epoch in range(NUM_EPOCHS):
    critic_losses = []
    gan_losses = []
    
    for batch_idx, (real, labels) in enumerate(trainloader):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        labels = labels.to(device)

        # Train Critic: max E[critic(real)] - E[critic(fake)] 
        # equivalent to minimizing the negative of that
        # min -[E[critic(real)] - E[critic(fake)]]
         
        for _ in range(CRITIC_ITERATIONS):
            random_image, random_image_tensor = load_random_image(folder_path)
            reshaped_tensor = random_image_tensor.repeat(batch_size, 1, 1, 1).reshape(batch_size, 100, 1, 1) + torch.randn((batch_size, 100, 1, 1)) 
            noise = reshaped_tensor.to(device)

            fake = gen(noise, labels)
            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)
            gp = gradient_penalty(critic, real, labels, fake, device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            critic_losses.append(loss_critic.item())  # Append critic loss to list

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]

        gen_fake = critic(fake, labels).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        gan_losses.append(loss_gen.item())  # Append GAN loss to list

        
        # Print losses occasionally
        # if batch_idx % 100 == 0 and batch_idx > 0:
            # print(
            #         f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(trainloader)} \
            #         Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            #     )
            # # Save and display generated images,
            # with torch.no_grad():
            #     fake = gen(noise, labels)
            #     torchvision.utils.save_image(fake[:batch_size], os.path.join('result4', f"fake_images_epoch_{epoch}.png"), normalize=True)

    # Log losses to wandb
    #wandb.log({"Critic Loss": sum(critic_losses) / len(critic_losses), "GAN Loss": sum(gan_losses) / len(gan_losses)}, step=epoch)

    # Print losses 
    print(
        f"Epoch [{epoch}/{NUM_EPOCHS}] \
        Loss D: {sum(critic_losses) / len(critic_losses):.4f}, \
        Loss G: {sum(gan_losses) / len(gan_losses):.4f}"
    )

    # Save and display generated images
    with torch.no_grad():
        fake = gen(noise, labels)
        torchvision.utils.save_image(fake[:batch_size], os.path.join('result_unpaired_1', f"fake_images_epoch_{epoch}.png"), normalize=True)
    
    if epoch % 5 == 0 and epoch > 0:

        torch.save(gen.state_dict(), os.path.join('result_unpaired_1', f"gen_weights_epoch_{epoch}.pt"))
        torch.save(critic.state_dict(), os.path.join('result_unpaired_1', f"critic_weights_epoch_{epoch}.pt"))

#wandb.finish()




# -*- coding: utf-8 -*-
"""classfier

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/classfier-028a614a-c0e2-4c66-a937-cf52093581a2.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20240414/auto/storage/goog4_request%26X-Goog-Date%3D20240414T180809Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3De9525f8d3f151cda6904181d9cb45c531161e7e2eefa4dd805ec1caffb35e8a521a908443bd8b85f7d121b8aa37726742278729df57fc378c7b2b0fdec7c7bb1e956b334e7379b27cc6d11bfd0556594241087ed7fab2d60b085c4f5430a1d22649f9d09cbd51f05bdd482bbaf891a12ea7af7e630cf8cbc1d239bfeffd30a16c09b9f407104d3c776e6a066711b94fb7e0fc6bf3acebf9c7409610dbe83b577395595246a100babb6ef93e8573b7880dbee617e67976da917e5a5ffaf31aa16a98ca5d92582a844f747064901f71d1c776d747a8934d589fae8f5da500a3b2702490acc02b01247c963879b49807281a132ad83406fa5cdb9b03218a322a04b
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import random

# Install efficientnet-pytorch
!pip install efficientnet-pytorch

# Import EfficientNet model
from efficientnet_pytorch import EfficientNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""# *Experiment Two*"""

class CustomDataset(Dataset):
    def __init__(self, img_folder, csv_file, transform=None):
        self.img_folder = img_folder
        self.csv_file = csv_file
        self.transform = transform

        # Read CSV file containing image filenames and one-hot encoded labels
        self.data_info = pd.read_csv(csv_file)

        # Number of items in the dataset
        self.data_len = len(self.data_info)

    def __getitem__(self, index):
        # Get image name from the dataframe
        img_name = os.path.join(self.img_folder, self.data_info.iloc[index, 0] + '.jpg') # Add file extension

        # Open image
        image = Image.open(img_name)

        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)

        # # Get one-hot encoded label from the dataframe
        label = self.data_info.iloc[index, 1:].values.astype(np.float32) # Convert labels to float32

        # # Convert one-hot encoded label to normal label
        label = np.argmax(label)
        return image, label

    def __len__(self):
        return self.data_len

transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5 for _ in range(3)], [0.5 for _ in range(3)]),])

# Example dataset instantiation
dataset = CustomDataset(img_folder='/kaggle/input/dataset-isisc-2016/Train_data-001/Train_data',
                        csv_file='/kaggle/input/dataset-isisc-2016/Assignment_4-20240412T123920Z-002/Assignment_4/Train/Train_labels.csv',
                        transform=transform)

train_indices, val_indices = train_test_split(
    list(range(len(dataset))),
    test_size=0.1,
    random_state=42
)

# Create training and validation datasets
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# Example dataloader creation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True,drop_last=True)

# Example dataset instantiation
testset = CustomDataset(img_folder='/kaggle/input/dataset-isisc-2016/Assignment_4-20240412T123920Z-002/Assignment_4/Test/Test_data/Test_data',
                        csv_file='/kaggle/input/dataset-isisc-2016/Assignment_4-20240412T123920Z-002/Assignment_4/Test/Test_labels.csv',
                        transform=transform)

# Example dataloader creation
test_loader = DataLoader(testset, batch_size=32, shuffle=False,drop_last=True)

class FineTunedEfficientNet(nn.Module):
    def __init__(self, model, num_classes):
        super(FineTunedEfficientNet, self).__init__()

        self.features = model

        self.features._conv_stem = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)  # Set bias to False

        # fully connected layer with a new sequence of linear layers
        self.features._fc = nn.Sequential(
            nn.Linear(model._fc.in_features, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),

            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.PReLU(),
            # nn.Dropout(p=0.5),

            nn.Linear(256, 64),
            nn.PReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Forward pass through the modified EfficientNet-B0
        x = self.features(x)

        return x


num_classes = 7
# Load the pre-trained EfficientNet-B0 model
pretrained_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)

# Create an instance of the fine-tuned model
fine_tuned_model = FineTunedEfficientNet(pretrained_model, 7)

# Move the fine-tuned model to the specified device
model = fine_tuned_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

num_epochs = 15

import wandb

# wandb.init(wandb.init(project='deeplens', name='effectnet'))


for epoch in range(num_epochs):
    model.train()
    total_correct_train = 0
    total_samples_train = 0
    running_loss_train = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device).float(), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predictions_train = torch.max(outputs, 1)
        total_correct_train += (predictions_train == labels).sum().item()
        total_samples_train += labels.size(0)
        running_loss_train += loss.item()

    train_accuracy = total_correct_train / total_samples_train
    train_loss = running_loss_train / len(train_loader)
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss)

    # Validation loop
    model.eval()
    with torch.no_grad():
        total_correct_val = 0
        total_samples_val = 0
        running_loss_val = 0.0

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predictions_val = torch.max(outputs, 1)
            total_correct_val += (predictions_val == labels).sum().item()
            total_samples_val += labels.size(0)
            running_loss_val += loss.item()

        val_accuracy = total_correct_val / total_samples_val
        val_loss = running_loss_val / len(val_loader)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        # Log metrics to wandb
#         wandb.log({
#             "Epoch" : epoch,
#             'Train Loss': train_loss,
#             'Train Accuracy': train_accuracy,
#             'Val Loss': val_loss,
#             'Val Accuracy': val_accuracy
#         })



        print(f'Epoch [{epoch+1}/{num_epochs}],'
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')


# wandb.finish()

# Plotting training and validation losses
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

torch.save(model.state_dict(), 'classification_model.pth')

path = '/kaggle/input/gan_weights/pytorch/gan-and-classification/2/classification_model.pth'
model.load_state_dict(torch.load(path, map_location=device))

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

# Assuming you have a test_loader for your test set
model.eval()
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device).float(), labels.to(device)

        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())

# Convert to NumPy array
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)



# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, np.argmax(all_probs, axis=1))

# Compute accuracy
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

print("Accuracy:", accuracy)

all_labels = all_labels.tolist()

len(all_labels)

conf_matrix

def load_random_image(folder_path, target_size=(10, 10)):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    random_image_filename = random.choice(image_files)
    random_image_path = os.path.join(folder_path, random_image_filename)

    image = Image.open(random_image_path)

    # Apply transformations
    transform_rand = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    image_tensor = transform_rand(image)

    return image, image_tensor


folder_path = "/kaggle/input/dataset-isisc-2016/Assignment_4-20240412T123920Z-002/Assignment_4/Test/Unpaired_test_sketch/Test_contours"

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, image_size, embed_size):
        super(Generator, self).__init__()
        self.embed = nn.Embedding(num_classes, embed_size)
        self.image_size = image_size

        self.conv1 = nn.ConvTranspose2d(channels_noise + embed_size, features_g * 16, kernel_size=4, stride=1, padding=0, bias=False)
        self.norm1 = nn.BatchNorm2d(features_g * 16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(features_g * 8)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(features_g * 4)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm4 = nn.BatchNorm2d(features_g * 2)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm5 = nn.BatchNorm2d(features_g)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.ConvTranspose2d(features_g, channels_img, kernel_size=4, stride=2, padding=1)
        self.upsample = nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=True)
        self.tanh = nn.Tanh()

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.relu5(self.norm5(self.conv5(x)))
        x = self.tanh(self.upsample(self.conv6(x)))
        return x

image_size = 64
# labels
num_classes = 7


# Instantiate the Generator class
gen = Generator(channels_noise=100, channels_img=3,
                features_g=64, num_classes=num_classes, image_size=image_size, embed_size=100).to(device)

# Load the saved weights

path = '/kaggle/input/gan_weights/pytorch/gan-and-classification/2/gen_weights_epoch_25.pt'
gen.load_state_dict(torch.load(path, map_location=device))

import torch
from torch.utils.data import Dataset, DataLoader

class GeneratedImageDataset(Dataset):
    def __init__(self, generated_images, labels):
        self.images = generated_images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]




# Generate 10 images for each class from 0 to 6
images = []
labels = []
for class_idx in all_labels:
    random_image, random_image_tensor = load_random_image(folder_path)
    # Generate fake images
    noise =  random_image_tensor.repeat(1, 1, 1, 1).reshape(1, 100, 1, 1).to(device)  + torch.randn((1, 100, 1, 1)).to(device)
    label = torch.tensor([class_idx] * 1, device=device)
    generated_images = gen(noise, label)
    images.extend(generated_images)
    labels.extend(label.tolist())

# Create a custom PyTorch dataset
dataset = GeneratedImageDataset(images, labels)

# Create a data loader
genloader = DataLoader(dataset, batch_size=4, shuffle=True)

import matplotlib.pyplot as plt
import numpy as np
import torchvision

# Define a function to display images and their labels
def show_images(dataset, num_images=5):
    # Set up a matplotlib figure
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))

    # Randomly select num_images samples from the dataset
    indices = np.random.choice(len(dataset), num_images, replace=False)

    for i, idx in enumerate(indices):
        # Get image and label
        image, label = dataset[idx]

        # Convert image tensor to numpy array and transpose channels
        image = np.transpose(image.cpu().detach().numpy(), (1, 2, 0))

        # Display image
        axes[i].imshow(image)
        axes[i].set_title(f"Label: {label}")

    plt.show()

# Assuming you have a PyTorch dataset called 'dataset'
# Show 5 random images from the dataset
show_images(dataset, num_images=5)

len(dataset)

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

# Assuming you have a test_loader for your test set
model = model.to(device)
model.eval()
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in genloader:
        inputs, labels = inputs.to(device).float(), labels.to(device)

        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())

# Convert to NumPy array
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)



# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, np.argmax(all_probs, axis=1))

# Compute accuracy
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

print("Accuracy:", accuracy)

conf_matrix

"""# Frechet Inception Distance (FID) and Inception Score"""

import torch
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.stats import entropy
import torchvision.transforms as T


# Load the pre-trained Inception model
inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).eval().cuda()

# Disable auxiliary logits
inception_model.aux_logits = False

# Define a function to calculate the Inception Score
def calculate_inception_score(dataloader, batch_size=32, resize=False, splits=10):
    """Computes the inception score of the generated images"""
    N = len(dataloader.dataset)

    # Get the predictions for all the images
    preds = get_inception_predictions(dataloader, inception_model, batch_size)

    #print(len(preds))

    # Split the predictions into equal chunks
    split_preds = torch.split(preds, N // splits)

    #print(len(split_preds))


    # Calculate the Inception Score for each chunk
    scores = torch.zeros(splits)
    for i in range(splits):
        chunk = split_preds[i]
        kl = chunk * (torch.log(chunk) - torch.log(torch.mean(chunk, dim=0, keepdim=True)))
        if kl.ndim == 1:
            scores[i] = torch.mean(kl)
        else:
            scores[i] = torch.mean(torch.sum(kl, dim=1))

    # Return the mean Inception Score
    return torch.mean(scores).item()

# Function to get the predictions from the Inception model

def get_inception_predictions(dataloader, model, batch_size=32):
    preds = []
    resize_transform = T.Resize(299)
    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.cuda()
            batch = resize_transform(batch)
            pred = model(batch)[0]
            preds.append(torch.softmax(pred, dim=-1).cpu())
    preds = torch.cat(preds, dim=0)
    return preds

# Calculate the Inception Score on test images
inception_score = calculate_inception_score(test_loader)
print(f'Inception Score: {inception_score}')

# Calculate the Inception Score on genreted images
inception_score = calculate_inception_score(genloader)
print(f'Inception Score: {inception_score}')

import torch.nn.functional as F

def calculate_inception_score(dataloader, model, splits=10):
    """Computes the inception score of the generated images"""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device)
            output = model.forward(batch)
            preds.append(F.softmax(output, dim=1).cpu().numpy())

    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)

inception_score, inception_std = calculate_inception_score(test_loader, fine_tuned_model, splits=10)
print(f"Inception Score: {inception_score:.2f} ± {inception_std:.2f}")

inception_score, inception_std = calculate_inception_score(genloader, fine_tuned_model, splits=10)
print(f"Inception Score: {inception_score:.2f} ± {inception_std:.2f}")

import numpy as np
from scipy.linalg import sqrtm


def calculate_fid(img1, img2):

    # Calculate the Frechet Inception Distance
    mean1, cov1 = img1.mean(axis=0), np.cov(img1, rowvar=False)

    #print(mean1.mean(),cov1.mean())
    mean2, cov2 = img2.mean(axis=0), np.cov(img2, rowvar=False)

    #print(mean2.mean(),cov2.mean())

    mean_diff = mean1.mean() - mean2.mean()
    mean_diff_squared = mean_diff * mean_diff

    covmean = np.sqrt((cov1.mean() * cov2.mean()))
    fid = mean_diff_squared + (cov1 + cov2 - 2 * covmean)
    print(fid)
    return float(fid)

import torch
from torchvision.transforms import Resize

def calculate_fid_score(real_loader, gen_loader, device):

    # Extract the features for the real and generated images
    real_features = []
    gen_features = []

    with torch.no_grad():
        for real_batch, gen_batch in zip(real_loader, gen_loader):
            # Extract the image data from the batches
            real_images = real_batch[0].to(device)
            gen_images = gen_batch[0].to(device)

            # Resize the images to 299x299 pixels
            real_images = Resize((299, 299))(real_images)
            gen_images = Resize((299, 299))(gen_images)

            # Extract the feature vectors
            real_feature = inception_model(real_images).cpu().numpy()
            gen_feature = inception_model(gen_images).cpu().numpy()

            real_features.append(real_feature)
            gen_features.append(gen_feature)

    # Concatenate the feature vectors
    real_features = np.concatenate(real_features, axis=0)
    gen_features = np.concatenate(gen_features, axis=0)

    # Calculate the FID score
    fid_score = calculate_fid(real_features, gen_features)

    return fid_score

# Calculate the FID score
fid_score = calculate_fid_score(test_loader, genloader, device)

print(f"Frechet Inception Distance (FID) score: {fid_score:.4f}")

import numpy as np
import matplotlib.pyplot as plt

p1 = '/kaggle/input/losses/losses_D.txt'
p2 ='/kaggle/input/losses/losses_G.txt'

import numpy as np
import matplotlib.pyplot as plt

# Load losses from text files
losses_D = np.loadtxt(p1)
losses_G = np.loadtxt(p)

# Plot losses in separate graphs
epochs = range(1, len(losses_D) + 1)

plt.figure(figsize=(12, 6))

# Plot Loss D
plt.subplot(2, 1, 1)
plt.plot(epochs, losses_D, color='blue')
plt.title('Discriminator Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

# Plot Loss G
plt.subplot(2, 1, 2)
plt.plot(epochs, losses_G, color='red')
plt.title('Generator Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.show()