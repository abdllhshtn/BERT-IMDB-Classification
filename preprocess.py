import os
import torch
from torch.utils.data import TensorDataset, DataLoader




DATA = "aclImdb"
positive_path = os.path.join(DATA,"train","pos")
positive_files = os.listdir(positive_path)
negative_path= os.path.join(DATA,"train","neg")
negative_files= os.listdir(negative_path)
positive_files = positive_files[:1000]
negative_files = negative_files[:1000]


texts=[]
labels=[]

for file_name in positive_files:
    file_path=os.path.join(positive_path,file_name)

    file=open(file_path,"r",encoding="utf-8")
    content=file.read()
    file.close()

    texts.append(content)
    labels.append(1)


for file_name in negative_files:
    file_path=os.path.join(negative_path,file_name)

    file=open(file_path,"r",encoding="utf-8")
    content=file.read()
    file.close()

    texts.append(content)
    labels.append(0)


tokenized_texts=[]

for text in texts:
    tokens=text.lower().split()
    tokenized_texts.append(tokens)

unique_words=set()
for tokens in tokenized_texts:
    for token in tokens:
        unique_words.add(token)

vocab = {"[PAD]": 0,"[UNK]": 1}
for index,word in enumerate(unique_words,start=2):
    vocab[word]=index

input_ids=[]
for tokens in tokenized_texts:
    review_ids=[]
    for token in tokens:
        review_ids.append(vocab.get(token,vocab["[UNK]"]))
    input_ids.append(review_ids)



MAX_LENGTH = 128
padded_input_ids = []
attention_masks = []
for review_ids in input_ids:
    if len(review_ids) > MAX_LENGTH:
        padded_review = review_ids[:MAX_LENGTH]
        attention_mask = [1] * MAX_LENGTH
    else:
        padding_amount = MAX_LENGTH - len(review_ids)
        padded_review = review_ids + [0] * padding_amount
        attention_mask = [1] * len(review_ids) + [0] * padding_amount
    padded_input_ids.append(padded_review)
    attention_masks.append(attention_mask)


input_ids = torch.tensor(padded_input_ids)
attention_masks = torch.tensor(attention_masks)
labels = torch.tensor(labels)
dataset = TensorDataset(input_ids, attention_masks, labels)
BATCH_SIZE = 32
dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

vocab_size = len(vocab)
embedding_dim = 64


