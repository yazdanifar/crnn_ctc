import logging

from PIL import Image
from torch.utils.data import Dataset
import os
import evaluate
from transformers import VisionEncoderDecoderModel
import torch
from transformers import TrOCRProcessor
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm


class CarDataset(Dataset):
    def __init__(self, root_dir, processor, max_target_length=8, train=True):
        self.root_dir = root_dir
        self.processor = processor
        self.max_target_length = max_target_length
        self.split = 'train' if train else 'test'
        self.data = self.load_data()

    def load_data(self):
        data = []
        a_dir_image = os.path.join(self.root_dir, 'CAR-A', f'a_{self.split}_images')
        a_gt = os.path.join(self.root_dir, 'CAR-A', f'a_{self.split}_gt.txt')
        b_dir_image = os.path.join(self.root_dir, 'CAR-B', f'b_{self.split}_images')
        b_gt = os.path.join(self.root_dir, 'CAR-B', f'b_{self.split}_gt.txt')
        for (gt_file, image_dir) in [(a_gt, a_dir_image), (b_gt, b_dir_image)]:
            with open(gt_file, 'r') as f:
                lines = f.read().splitlines()
            for line in lines:
                image_name, label = line.split()
                image_path = os.path.join(image_dir, image_name)
                data.append((image_path, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, text = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding



logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("trocr.log")
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
train_dataset = CarDataset(root_dir='./orand_car_2014', train=True, processor=processor)

eval_dataset = CarDataset(root_dir='./orand_car_2014', train=False, processor=processor)


train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-stage1")
model.to(device)

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

cer_metric = evaluate.load("cer")


def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer


optimizer = AdamW(model.parameters(), lr=5e-5)
for epoch in range(100):  # loop over the dataset multiple times
    # train
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_dataloader):
        # get the inputs
        for k, v in batch.items():
            batch[k] = v.to(device)

        # forward + backward + optimize
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    logger.info(f"Loss after epoch {epoch}: {train_loss / len(train_dataloader):.5f}")

    # evaluate
    model.eval()
    valid_cer = 0.0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            # run batch generation
            outputs = model.generate(batch["pixel_values"].to(device), max_length=8)
            # compute metrics
            cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
            valid_cer += cer

    logger.info(f"Test CER: {valid_cer / len(eval_dataloader):.5f}")
