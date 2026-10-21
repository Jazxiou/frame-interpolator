# My own utilities, some inspired by Jon Macey's workshops.
import io
import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from IPython.display import HTML, display
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class FrameInterpolation(Dataset):
    """
    The dataset structure for the Frame Interpolator model.
    """

    # I decided to go with PIL (not cv2) for consistency with the rest of the project.
    def __init__(self, df, augment=False, target_size=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing columns "Image 1 (Input)",
            "Image 3 (Input)", and "Image 2 (output)".
            augment (bool, optional): If true, applies augmentations. Default is False.
            target_size (tuple, optional): Tuple (H, W) to resize images to. Default is None.
        """
        super().__init__()
        self.img1_paths = df["Image 1 (Input)"].values
        self.img3_paths = df["Image 3 (Input)"].values
        self.img2_paths = df["Image 2 (Output)"].values
        self.augment = augment
        self.target_size = target_size

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: ((img1, img3), img2) where all elements are torch.Tensors.
            Input is grouped as a tuple for the model.
        """
        # Load Images
        img1 = Image.open(self.img1_paths[idx]).convert("RGB")
        img3 = Image.open(self.img3_paths[idx]).convert("RGB")
        img2 = Image.open(self.img2_paths[idx]).convert("RGB")

        # Resize if needed
        if self.target_size:
            img1 = img1.resize(self.target_size)
            img3 = img3.resize(self.target_size)
            img2 = img2.resize(self.target_size)

        # Convert to tensor
        t_img1 = TF.to_tensor(img1)
        t_img3 = TF.to_tensor(img3)
        t_img2 = TF.to_tensor(img2)

        # Data Augmentation (only runs if class "augment" is true)
        if self.augment:
            if random.random() > 0.5:  # Random Horizontal Flip (50% chance)
                t_img1 = TF.hflip(t_img1)
                t_img3 = TF.hflip(t_img3)
                t_img2 = TF.hflip(t_img2)

            if random.random() > 0.5:  # Random Rotation (50% chance)
                angle = random.uniform(-15, 15)
                t_img1 = TF.rotate(t_img1, angle)
                t_img3 = TF.rotate(t_img3, angle)
                t_img2 = TF.rotate(t_img2, angle)

            if random.random() > 0.5:  # Random Brightness (50% chance)
                brightness = random.uniform(0.9, 1.1)
                t_img1 = TF.adjust_brightness(t_img1, brightness)
                t_img3 = TF.adjust_brightness(t_img3, brightness)
                t_img2 = TF.adjust_brightness(t_img2, brightness)

        return (t_img1, t_img3), t_img2

    def __len__(self):
        return len(self.img1_paths)


# ---- Training Tools


def get_batch_accuracy(output, y, accuracy_metric, N):
    """
    Calculates the accuracy for a single batch.

    Args:
        output (torch.Tensor): Model predictions.
        y (torch.Tensor): Ground truth.
        accuracy_metric (callable): Function to compute accuracy score.
        N (int): The number of samples in the dataset.

    Returns:
        float: The batch's total epoch accuracy.
    """
    score = accuracy_metric(output, y)
    correct = score.item() * output.size(0)
    return correct / N


def train(model, dataloader, optimizer, loss_fn, accuracy_metric, device, N):
    """
    Trains the model for one epoch over the dataset.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (DataLoader): The iterator for the dataset.
        optimizer (torch.optim.Optimizer): The optimizer function.
        loss_fn (torch.nn.Module): The loss function.
        accuracy_metric (callable): Function to compute accuracy score.
        device (torch.device): The active device to use.
        N (int): The number of samples in the dataset.

    Returns:
        tuple: (Average epoch loss, Average epoch accuracy)
    """
    model.train()
    running_loss = 0.0
    accuracy = 0.0
    for batch in dataloader:
        (img1, img3), img2 = batch
        img1, img3, img2 = img1.to(device), img3.to(device), img2.to(device)

        preds = model(img1, img3)

        loss_l1 = loss_fn(preds, img2)
        # SSIM returns 1 for perfect, so I did 1 - SSIM to make it 0 for perfect.
        loss_ssim = 1 - accuracy_metric(preds, img2)
        # L1 for colours, SSIM for sharpness
        loss = (0.8 * loss_l1) + (0.2 * loss_ssim)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img2.size(0)
        accuracy += get_batch_accuracy(preds, img2, accuracy_metric, N)
    return running_loss / N, accuracy


@torch.no_grad()
def validate(model, dataloader, loss_fn, accuracy_metric, device, N):
    """
    Evaluates the model's performance on a validation or test dataset.
    Disables gradient calculation and ensures the model is in evaluation mode.

    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (DataLoader): The iterator for the dataset.
        loss_fn (torch.nn.Module): The loss function.
        accuracy_metric (callable): Function to compute accuracy score.
        device (torch.device): The active device to use.
        N (int): The number of samples in the dataset.

    Returns:
        tuple: (Average validation epoch loss, Average validation epoch accuracy)
    """
    model.eval()
    running_loss = 0.0
    accuracy = 0.0
    for batch in dataloader:
        (img1, img3), img2 = batch
        img1, img3, img2 = img1.to(device), img3.to(device), img2.to(device)

        preds = model(img1, img3)

        loss_l1 = loss_fn(preds, img2)
        loss_ssim = 1 - accuracy_metric(preds, img2)
        loss = (0.8 * loss_l1) + (0.2 * loss_ssim)

        running_loss += loss.item() * img2.size(0)
        accuracy += get_batch_accuracy(preds, img2, accuracy_metric, N)
    return running_loss / N, accuracy


# ---- Core Tools ----


def preprocess_image(content, target_size, device):
    """
    Converts raw image (from widgets) to a tensor.

    Args:
        content (bytes): Raw image byte data.
        target_size (tuple): Desired (H, W)for the image.
        device (torch.device): The active device to use.

    Returns:
        torch.Tensor: Preprocessed image tensor with batch dim (1, C, H, W).
    """
    img = Image.open(io.BytesIO(content)).convert("RGB")
    if target_size:
        img = img.resize(target_size)
    return TF.to_tensor(img).unsqueeze(0).to(device)


def load_tensor_from_path(path, target_size=None):
    """
    Loads an image from a file path into a tensor.

    Args:
        path (str): File path to the image.
        target_size (tuple, optional): (H, W) to resize the image.

    Returns:
        torch.Tensor: Image tensor with batch dim (1, C, H, W)
    """
    img = Image.open(path).convert("RGB")
    if target_size:
        img = img.resize(target_size)
    return TF.to_tensor(img).unsqueeze(0)


def display_image(tensor):
    """
    Converts a tensor (N, C, H, W) or (C, H, W) to a Numpy array (H, W, C).
    Handles detaching, CPU move, clamping, and dimension permutation.

    Args:
        tensor (torch.Tensor): An image tensor.

    Returns:
        np.ndarray: The image array.
    """
    img = tensor.cpu().detach()
    if img.dim() == 4:
        img = img.squeeze(0)  # Remove batch dimension
    img = img.clamp(0, 1).permute(1, 2, 0)
    return img.numpy()


# ---- Visualisation Tools ----


def plot_dataset_samples(df, num_rows=4):
    """
    Visualises raw data from a path(dataframe).

    Args:
        df (pd.DataFrame): The dataframe containing image paths.
        num_rows (int, optional): Number of rows/samples to display. Default is 4.
    """
    plt.figure(figsize=(10, 2.5 * num_rows))

    for idx in range(num_rows):
        # Load paths
        path1 = df["Image 1 (Input)"][idx]
        path2 = df["Image 2 (Output)"][idx]
        path3 = df["Image 3 (Input)"][idx]
        # Convert to tensor
        t1 = load_tensor_from_path(path1)
        t2 = load_tensor_from_path(path2)
        t3 = load_tensor_from_path(path3)
        # Plot
        images = [t1, t2, t3]
        titles = ["Image 1 (Input)", "Image 2 (Ground Truth)", "Image 3 (Input)"]

        for col, (image, title) in enumerate(zip(images, titles)):
            plt.subplot(num_rows, 3, 3 * idx + col + 1)
            plt.imshow(display_image(image))
            if idx == 0:  # Only title for the top row
                plt.title(title, fontsize=10)
            plt.axis("off")
    plt.show()


def plot_results(model, dataloader, device, num_samples=4):
    """
    Visualises model predictions against ground truth.
    Top Row: Start, Ground Truth, End
    Bottom Row: Start, AI Output, End

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): Dataloader to fetch a batch from.
        device (torch.device): The active device to use.
        num_samples (int, optional): Number of samples to plot. Default is 4.
    """
    model.eval()
    with torch.no_grad():
        (img1, img3), img2 = next(iter(dataloader))
        output = model(img1.to(device), img3.to(device))

    fig, ax = plt.subplots(nrows=num_samples * 2, ncols=3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        row_real = i * 2
        row_output = i * 2 + 1

        # The Ground Truth
        # Input 1
        ax[row_real, 0].imshow(display_image(img1[i]))
        ax[row_real, 0].set_title("Image 1 (Input)")
        ax[row_real, 0].axis("off")
        # Real Image
        ax[row_real, 1].imshow(display_image(img2[i]))
        ax[row_real, 1].set_title("Image 2 (Ground Truth)")
        ax[row_real, 1].axis("off")
        # Input 3
        ax[row_real, 2].imshow(display_image(img3[i]))
        ax[row_real, 2].set_title("Image 3 (Input)")
        ax[row_real, 2].axis("off")

        # The AI Output
        # Input 1
        ax[row_output, 0].imshow(display_image(img1[i]))
        ax[row_output, 0].axis("off")
        # AI Image
        ax[row_output, 1].imshow(display_image(output[i]))
        ax[row_output, 1].set_title("Image 2 (AI Output)")
        ax[row_output, 1].axis("off")
        # Input 3
        ax[row_output, 2].imshow(display_image(img3[i]))
        ax[row_output, 2].axis("off")
    plt.show()


# ---- Interactive Tools ----


def create_gif(model, device, dataloader=None, img1=None, img3=None):
    """
    Creates a gif from either a dataloader or images uploaded in the demo.

    Args:
        model(torch.nn.Module): The trained model.
        device(torch.device): The active device to use.
        dataloader (DataLoader, optional): Gets a random batch from here.
        img1, img3 (torch.Tensor, optional): Start and end frames.

    Returns:
        IPython.display.HTML: A JSHTML animation/gif.
    """
    model.eval()
    with torch.no_grad():
        if dataloader is not None:
            (batch_img1, batch_img3), _ = next(iter(dataloader))
            # Add batch dimension back
            input1 = batch_img1[0].unsqueeze(0).to(device)
            input3 = batch_img3[0].unsqueeze(0).to(device)

        elif img1 is not None and img3 is not None:
            input1 = img1.to(device) if img1.dim() == 4 else img1.unsqueeze(0).to(device)
            input3 = img3.to(device) if img3.dim() == 4 else img3.unsqueeze(0).to(device)

        else:
            print("Error: Provide either 'dataloader' OR 'img1' and 'img3'")
            return None

        img2 = model(input1, input3)  # Run Model

        frames = [display_image(x) for x in [input1, img2, input3]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    im = ax.imshow(frames[0])
    plt.close()

    def update_gif_frame(frame):
        im.set_array(frame)
        return [im]

    gif = animation.FuncAnimation(fig, update_gif_frame, frames=frames, interval=200)
    return HTML(gif.to_jshtml())


def process_widget(widget, target_size, device):
    """
    Checks widget, extracts content, and returns a ready-to-use tensor.

    Args:
        widget (ipywidgets.Widget): The file upload widget.
        target_size (tuple): Resize dimensions (H, W).
        device (torch.device): The active device to use.

    Returns:
        torch.Tensor or None: The image tensor, or None if widget is empty.
    """
    if not widget.value:
        return None
    return preprocess_image(widget.value[0]["content"], target_size, device)


def generate_demo(out, upload_1, upload_2, target_size, device, model):
    """
    Generates an image from custom input.

    Args:
        out (ipywidgets.Output): The output widget to render into.
        upload_1, upload_2 (ipywidgets.FileUpload): Widget for start and end frame.
        target_size (tuple): Resize dimensions.
        device (torch.device): The active device to use.
        model (torch.nn.Module): The trained model.

    Returns:
        tuple: The visualised prediction.
    """
    with out:
        out.clear_output()

        # Preprocess
        img1 = process_widget(upload_1, target_size, device)
        img3 = process_widget(upload_2, target_size, device)

        # Check if images were uploaded
        if img1 is None or img3 is None:
            print("Please upload both inputs first!")
            return None, None

        # Run Model
        model.eval()
        with torch.no_grad():
            img2 = model(img1, img3)

        # Postprocess
        images = [display_image(x) for x in [img1, img2, img3]]
        titles = ["Frame 1 (Start)", "Frame 2 (AI Ouput)", "Frame 3 (End)"]

        # Display
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i, (image, title) in enumerate(zip(images, titles, strict=True)):
            ax[i].imshow(image)
            ax[i].set_title(title)
            ax[i].axis("off")
        plt.show()

        return img1, img3
