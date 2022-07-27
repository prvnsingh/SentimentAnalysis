import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AdamW, DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split

from dataset import AirlineDataset

df = pd.read_csv("airline_sentiment_analysis.csv")
# df = df.applymap(str,axis=1)
data = df.to_numpy()

labels = data[1:, 1:2].reshape(-1)
text = data[1:, 2:].reshape(-1)
labels[labels == "negative"] = 0
labels[labels == "positive"] = 1
# text = np.array2string(text)
train_texts, test_texts, train_labels, test_labels = train_test_split(text, labels, test_size=.2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_name = "distilbert-base-uncased"  # base model

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
train_texts = np.array_str(train_texts)
test_texts = np.array_str(test_texts)
# print(test_texts)
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = AirlineDataset(train_encodings, train_labels)
test_dataset = AirlineDataset(test_encodings, test_labels)

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset       # training dataset            # evaluation dataset
)

trainer.train()


# model = DistilBertForSequenceClassification.from_pretrained(model_name)
# model.to(device)
# model.train()
#
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# optim = AdamW(model.parameters(), lr=5e-5)
#
# num_epoch = 5
# for epoch in range(num_epoch):
#     for batch in train_loader:
#         optim.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#
#         loss = outputs[0]
#         loss.backward()
#         optim.step()
#
# model.eval()
#
result = model.predict(test_dataset)
print(result)
