import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch


def display_image_in_actual_size(img_tensor, B, dev):

    if B > 1:
        img_np = np.transpose(arrange_channel_dim_to_block_pixels(img_tensor, B, dev)[0].cpu().detach().numpy() +0.5, (1, 2, 0))
    else:
        img_np = np.transpose(img_tensor[0].cpu().detach().numpy() + 0.5, (1, 2, 0))

    dpi = mpl.rcParams['figure.dpi']
    height, width, depth = img_np.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    if depth == 3:
        ax.imshow(img_np, cmap='gray')
    elif depth == 1:
        ax.imshow(img_np[:, :, 0], cmap='gray')

    plt.show()


def plot_hist_of_rgb_image(image):
    image = int((image + 0.5)*255)
    #_ = plt.hist(image.ravel(), bins = 256, color = 'orange', )
    _ = plt.hist(image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
    _ = plt.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
    _ = plt.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Count')
    _ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.show()


def arrange_block_pixels_to_channel_dim(x, B, dev):
    C, H, W = x.shape[1], x.shape[2], x.shape[3]
    y = torch.empty(x.shape[0], C*(B**2), H//B, W//B, device=dev)
    for v in range(0, B, 1):
        for h in range(0, B, 1):
            indd = (v*B+h)*C
            y[:, indd:indd+C, :, :] = x[:, :, v::B, h::B]
    return y


def arrange_channel_dim_to_block_pixels(y, B, dev):
    C, H, W = y.shape[1], y.shape[2], y.shape[3]
    C = C//(B**2)
    H = H*B
    W = W*B
    x = torch.empty(y.shape[0], C, H, W, device=dev)
    for v in range(0, B, 1):
        for h in range(0, B, 1):
            indd = (v*B+h)*C
            x[:, :, v::B, h::B] = y[:, indd:indd+C, :, :]
    return x
