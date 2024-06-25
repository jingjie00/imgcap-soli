import os
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from torchvision import transforms
from PIL import Image
import numpy as np


types = ['flickr8k_images', 'R0.1S1', 'R0.1S50', 'R0.2S1', 'R0.2S50', 'R0.5S1', 'R0.5S50', 'R0.05S50', 'R0.5S1_GF500', 'R1S1_GF500']
base_path = './datasets/flickr8k'
image_filenames = ['667626_18933d713e.jpg', '3637013_c675de7705.jpg', '10815824_2997e03d76.jpg', '12830823_87d2654e31.jpg', '17273391_55cfc7d3d4.jpg']
extra_image = ['/flickr8k_images/19212715_20476497a3.jpg']

path_sets = [[f'{base_path}/{p}/{img}' for p in types] for img in image_filenames]


for p in path_sets:
    p.extend(f'{base_path}/{e}' for e in extra_image)



# Load encoder model
model = VisionEncoderDecoderModel.from_pretrained("atasoglu/vit-gpt2-flickr8k")
feature_extractor = ViTImageProcessor.from_pretrained("atasoglu/vit-gpt2-flickr8k")
encoder = model.encoder
# load state dict
state =torch.load('./siamese/saved/vit_siamese_best_val_loss.pt', map_location='cpu')
encoder.load_state_dict(state["state_dict"])

encoder.eval()  # set model to evaluation mode

# Ensure model is on the correct device
device = torch.device('cpu')  # change to 'cuda' if using GPU


encoder = encoder.to(device)

def pairwise_distance(tensor1, tensor2):
    """
    Calculate the Euclidean distance between two 2D tensors.

    Parameters:
    tensor1 (torch.Tensor): The first 2D tensor.
    tensor2 (torch.Tensor): The second 2D tensor.

    Returns:
    float: The Euclidean distance between the two tensors.
    """
    # Ensure the tensors are 2D
    assert tensor1.dim() == 2 and tensor2.dim() == 2, "Both tensors must be 2D"

    # Calculate the Euclidean distance
    distance = torch.norm(tensor1 - tensor2)
    return distance

# Function to compute distance matrix for a set of images
def compute_distance_matrix(image_paths):

    images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
    images = feature_extractor(images=images, return_tensors="pt").pixel_values
    images = images.to(device)

    with torch.no_grad():  # disable gradient computation for evaluation
        embeddings = encoder(images).last_hidden_state

    num_images = embeddings.shape[0]
    distance_matrix = np.zeros((num_images, num_images))
    
    for i in range(num_images):
        for j in range(num_images):
            distance_matrix[i, j] = pairwise_distance(embeddings[i], embeddings[j]).item()
    
    return distance_matrix

# Compute distance matrices for all sets
distance_matrices = [compute_distance_matrix(paths) for paths in path_sets]




# Compute the average distance matrix
average_distance_matrix = np.mean(distance_matrices, axis=0)

# Function to plot a distance matrix with color
def plot_distance_matrix(matrix, title, labels, output_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(matrix, cmap='viridis', interpolation='nearest')
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val:.4g}', ha='center', va='center', color='white' if val < np.max(matrix)/2 else 'black')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

# Create directory to save the images
output_dir = 'distance_matrices'
os.makedirs(output_dir, exist_ok=True)

# Plot and save individual distance matrices
for idx, matrix in enumerate(distance_matrices):
    output_path = os.path.join(output_dir, f'distance_matrix_{idx + 1}.png')
    plot_distance_matrix(matrix, f'Matrix {idx + 1}', types + ['extra'], output_path)

# Plot and save the average distance matrix
output_path = os.path.join(output_dir, 'average_distance_matrix.png')
plot_distance_matrix(average_distance_matrix, 'Average Matrix', types + ['flickr_extra'], output_path)

print("All distance matrices have been saved successfully.")
