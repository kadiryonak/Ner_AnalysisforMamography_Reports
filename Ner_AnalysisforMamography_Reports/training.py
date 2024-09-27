import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Mixed precision training and evaluation function
def train_and_evaluate_model_with_mixed_precision(model, data_loader, optimizer, tag_to_ix, epochs=3150):
    model.train()
    loss_function = nn.CrossEntropyLoss(ignore_index=tag_to_ix["<PAD>"])
    scaler = GradScaler(enabled=False)

    f1_scores = []  # F1 skorlarını tutacak liste

    for epoch in range(epochs):
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for sentences, tags in data_loader:
            optimizer.zero_grad()

            with autocast(enabled=False):  # CPU'da autocast devre dışı
                outputs = model(sentences)
                outputs = outputs.view(-1, outputs.shape[-1])  # (batch_size * seq_len, tagset_size)
                tags = tags.view(-1)  # (batch_size * seq_len)

                loss = loss_function(outputs, tags)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            mask = tags != tag_to_ix["<PAD>"]
            preds = preds[mask]
            tags = tags[mask]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(tags.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        f1_scores.append(f1)  # F1 skorunu listeye ekle

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, F1 Score: {f1:.4f}")

    # Show F1 scores on a graph after training is complete 
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), f1_scores, label="F1 Score", color='b')
    plt.title("F1 Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.show(block=True)
