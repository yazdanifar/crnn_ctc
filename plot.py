import torch
import matplotlib.pyplot as plt


def plot_batch(writer, image, decoded, mean, std, global_step):
    random_numbers = torch.randint(low=0, high=image.size(0), size=(10,))
    fig = plt.figure(figsize=(10, 3))
    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1)
        image_to_plot = image[random_numbers[i]].cpu().numpy().transpose(1, 2, 0) * std + mean
        plt.imshow(image_to_plot, cmap='gray')
        ax.set_title(decoded[random_numbers[i]])  # Convert label to text
    writer.add_figure('predictions', fig, global_step=global_step)
