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
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = AirlineDataset(train_encodings, train_labels)
test_dataset = AirlineDataset(test_encodings, test_labels)

model = DistilBertForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
optim = AdamW(model.parameters(), lr=5e-5)

num_epoch = 5
for epoch in range(num_epoch):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()

result = model.predict(test_dataset)
print(result)
