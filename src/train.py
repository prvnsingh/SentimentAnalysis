import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from dataset import AirlineDataset

# data
df = pd.read_csv("../airline_sentiment_analysis.csv") # airline review data


# converting dataframe to numpy array to reshape and preprocess
data = df.to_numpy()
labels = data[1:, 1:2].reshape(-1)
text = data[1:, 2:].reshape(-1)
labels[labels == "negative"] = 0
labels[labels == "positive"] = 1

# splitting into train and test set
train_texts, test_texts, train_labels, test_labels = train_test_split(text, labels, test_size=.2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# get the base model from hugging face
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# initializing tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, labels=2)

train_texts = np.array_str(train_texts)
test_texts = np.array_str(test_texts)

train_encodings = tokenizer.encode_plus(train_texts, add_special_tokens=True, truncation=True, padding=True)
test_encodings = tokenizer.encode_plus(test_texts, add_special_tokens=True, truncation=True, padding=True)

train_dataset = AirlineDataset(train_encodings, train_labels)
test_dataset = AirlineDataset(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir='../results',  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='../logs',  # directory for storing logs
    logging_steps=10,
    label_names=[0,1]
)

model = AutoModelForSequenceClassification.from_pretrained(model_name,id2label={0: 'negative', 1: 'positive'})

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics
)
trainer.train()


# test_trainer = Trainer(model)
#
# # print(test_dataset.__getitem__(0))
# pred,_,_ = test_trainer.predict(test_dataset)
# print(np.argmax(pred,axis = 1))