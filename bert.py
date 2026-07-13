import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import dataloader, vocab_size
from model import MiniBERT


model = MiniBERT(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_input_ids, batch_attention_mask, batch_labels in dataloader:
        outputs = model.forward(batch_input_ids,batch_attention_mask)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == batch_labels).sum().item()
        total += batch_labels.size(0)

    accuracy = correct / total

    print("Epoch:", epoch + 1)
    print("Loss:", total_loss)
    print("Accuracy:", accuracy)