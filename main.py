import os
import argparse  # Import argparse for command-line argument parsing

import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from data import CarDataset
from decoders import best_path, beam_search_batch
from evaluation import calculate_cer_for_batch
from networks import CRNN
from tensorboardX import SummaryWriter
import numpy as np
import random
from plot import plot_batch


def test_eval(crnn, decoder, test_loader):
    cer = 0
    crnn.eval()
    with torch.no_grad():
        for num_batches, (image, target_lengths, target) in enumerate(test_loader):
            image, target_lengths, target = image.cuda(), target_lengths.cuda(), target.cuda()
            out = crnn(image)
            decoded = decoder(out)
            cer += calculate_cer_for_batch(target_lengths, target, decoded)
    crnn.train()
    return cer / len(test_loader)


def main(args):
    root_dir = args.root_dir
    epochs = args.epochs
    training_mean = args.training_mean
    training_std = args.training_std
    decoder_strategy = args.decoder_strategy
    batch_size = args.batch_size
    seed = args.seed
    conv_channels = args.conv_chans
    lr = args.lr
    beam_width = args.beam_width

    transform = transforms.Compose([
        transforms.Resize((54, 172)),
        transforms.ToTensor(),
        transforms.Normalize(mean=training_mean, std=training_std)
    ])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset = CarDataset(root_dir, transform=transform, train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = CarDataset(root_dir, transform=transform, train=False)
    test_loader = DataLoader(test_dataset, batch_size=3 * batch_size, shuffle=False)

    crnn = CRNN(rnn_hidden_size=128, rnn_num_layers=2, channels=conv_channels).cuda()
    ctc_loss = nn.CTCLoss(blank=10, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(crnn.parameters(), lr=lr)
    decoder = best_path if decoder_strategy == 'greedy' else lambda x: beam_search_batch(x, beam_width)
    writer = SummaryWriter()

    total_batches = len(train_loader)
    log_interval = total_batches // 10

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_cer = 0
        for num_batches, (image, target_lengths, target) in enumerate(train_loader):
            image, target_lengths, target = image.cuda(), target_lengths.cuda(), target.cuda()
            optimizer.zero_grad()
            out = crnn(image)
            input_lengths = torch.full(size=(out.shape[1],), fill_value=out.shape[0], dtype=torch.long, device='cuda')
            loss = ctc_loss(out, target, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            decoded = decoder(out)
            epoch_cer += calculate_cer_for_batch(target_lengths, target, decoded)
            if num_batches % log_interval == 0 or num_batches == total_batches - 1:
                print(f'Epoch: {epoch}/{epochs - 1} - Batch: {num_batches}/{total_batches - 1}'
                      f' - Train Loss: {epoch_loss / (num_batches + 1):.5f}'
                      f' - Train CER: {epoch_cer / (num_batches + 1):.5f}')
                global_step = epoch * total_batches + num_batches
                writer.add_scalar('Train Loss', epoch_loss / (num_batches + 1), global_step)
                writer.add_scalar('Train CER', epoch_cer / (num_batches + 1), global_step)
                plot_batch(writer, image, decoded, training_mean, training_std, global_step)

        test_cer = test_eval(crnn, decoder, test_loader)
        print(f'Epoch: {epoch}/{epochs - 1} - Test CER: {test_cer:.5f}')
        global_step = (epoch + 1) * total_batches - 1
        writer.add_scalar('Test CER', test_cer, global_step)


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="CRNN Training with Adjustable Constants")
    parser.add_argument("--root_dir", type=str, default='./orand_car_2014', help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--training_mean", type=float, default=0.6277, help="Mean for data normalization")
    parser.add_argument("--training_std", type=float, default=0.1645, help="Standard deviation for data normalization")
    parser.add_argument("--decoder_strategy", type=str, default='greedy', choices=['greedy', 'beam'],
                        help="Decoder strategy")
    parser.add_argument("--beam_width", type=int, default=25, help="Maximum Candidate Beams")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--conv_chans", type=list, default=[32, 32, 64], help="Convolutional Channels")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")

    args = parser.parse_args()

    main(args)
