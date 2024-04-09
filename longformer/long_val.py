# -*- coding: utf-8 -*-
"""Copy of longformer+pytorchlightning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11EefNhlprSa22PCCYyxT_XU8cpWsc8YO
"""

# from google.colab import drive
# drive.mount('/content/drive')

# !pip install transformers[torch]
# !pip install lightning

# !pip install transformers
# !pip install datasets
# !pip install wandb

import torch
import torchtext
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, BertModel
from transformers import AutoTokenizer, LongformerModel, LongformerTokenizer
import pytorch_lightning as pl
from tqdm.auto import tqdm
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig

# def tokenize_and_pad_text(text, max_length,tokenizer):
#     tokens = tokenizer.tokenize(text)
#     if len(tokens) > max_length - 2:  # for [CLS] and [SEP] tokens
#         tokens = tokens[:max_length - 2]

#     # Add special tokens ([CLS] and [SEP]) and pad to max_length
#     input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
#     input_ids += [0] * (max_length - len(input_ids))  # Pad with zeros
#     return input_ids

def tokenize_and_pad_text(text, max_length):
    encoded = tokenizer.encode_plus(
        text,
        max_length=max_length,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoded['input_ids']
    return input_ids

class LongformerClassifier(pl.LightningModule):
    def __init__(self, num_classes, tokenizer,model_name="allenai/longformer-base-4096"):
        super().__init__()
        self.longformer = LongformerModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.longformer.config.hidden_size, num_classes)
        self.tokenizer=tokenizer
        
    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        logits = self.classifier(last_hidden_state[:, 0, :])  # Use CLS token for classification
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.log('train_loss', loss)
#        print("train_loss ",loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.log('val_loss', loss)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels) / len(preds)
        self.log('val_acc', acc)
#        print("validation_acc: ",acc)
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)

if __name__ == '__main__':
        
    mono_train_data = pd.read_json('data/SemEval8/SubtaskA/subtaskA_train_monolingual.jsonl', lines=True)
    mono_dev_data = pd.read_json('data/SemEval8/SubtaskA/subtaskA_dev_monolingual.jsonl', lines=True)

    model_name = "allenai/longformer-base-4096"
    tokenizer = LongformerTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_sequence_length = 4096

    # mono_train_data = mono_train_data[:500]
    mono_dev_data = mono_dev_data

    # mono_train_data['tokenized_text'] = mono_train_data['text'].apply(lambda x: tokenize_and_pad_text(x, max_sequence_length))
    mono_dev_data['tokenized_text'] = mono_dev_data['text'].apply(lambda x: tokenize_and_pad_text(x, max_sequence_length,tokenizer))

    # train_inputs = torch.stack([torch.tensor(arr).detach() for arr in mono_train_data['tokenized_text'].values])
    dev_inputs = torch.stack([torch.tensor(arr).detach() for arr in mono_dev_data['tokenized_text'].values])

    # train_labels = torch.tensor(mono_train_data['label'].values)
    dev_labels = torch.tensor(mono_dev_data['label'].values)

    # Create TensorDatasets
    # train_dataset = TensorDataset(train_inputs, train_labels)
    dev_dataset = TensorDataset(dev_inputs, dev_labels)

    batch_size = 1

    # Create DataLoaders
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 2
    model = LongformerClassifier(num_classes=num_classes, model_name="allenai/longformer-base-4096",tokenizer=tokenizer)
    model = model.to(device)
    # trainer = pl.Trainer(max_epochs=3, gpus=1)
    # trainer = pl.Trainer(max_epochs=3, accelerator="auto")

    # trainer.fit(model, train_dataloader, dev_dataloader)

    total_correct = 0
    total_examples = 0
    model.load_state_dict(torch.load("/home2/pradhuman.t/lightning_logs/version_991342/checkpoints/epoch=1-step=25000.ckpt")['state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    # Initialize variables to keep track of correct predictions and total examples
    total_correct = 0
    total_examples = 0

    # Set the model to evaluation mode
    model.eval()

    # Iterate through the validation dataloader
    for batch in tqdm(dev_dataloader):
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Calculate attention mask
        attention_mask = (input_ids != tokenizer.pad_token_id).to(device)

        # Forward pass through the model
        with torch.no_grad():  # Make sure to use no_grad to avoid unnecessary gradient computation during validation
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Calculate predictions and update counts
        preds = torch.argmax(logits, dim=1)
        correct = torch.sum(preds == labels)
        total_correct += correct.item()
        total_examples += len(labels)

    # Calculate validation accuracy
    validation_accuracy = total_correct / total_examples

    # Print or use the validation accuracy as needed
    print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

