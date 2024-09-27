import torch
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
# Training And Evaluation Function
def train_and_evaluate_model_with_mixed_precision(model, data_loader, optimizer, epochs=3150):
    model.train()
    loss_function = nn.CrossEntropyLoss(ignore_index=tag_to_ix["<PAD>"])
    scaler = GradScaler(enabled=False)  # CPU'da 'enabled=False' ile ekledik

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
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, F1 Score: {f1:.4f}")


# 5. Curriculum Learning
def difficulty_level(report):
    return len(report.split())

def split_data_by_difficulty(bio_data):
    bio_data_sorted = sorted(bio_data, key=lambda x: difficulty_level(x[0]))  # Zorluk seviyesine göre sıralandı
    split_1 = int(len(bio_data_sorted) * 0.33)
    split_2 = int(len(bio_data_sorted) * 0.66)

    # Basit, orta ve zor veriler
    easy_data = bio_data_sorted[:split_1]
    medium_data = bio_data_sorted[split_1:split_2]
    hard_data = bio_data_sorted[split_2:]
    
    return easy_data, medium_data, hard_data

def train_model_with_curriculum_learning(model, easy_data, medium_data, hard_data, word_to_ix, tag_to_ix, optimizer, epochs=3150):
    for stage, data in enumerate([easy_data, medium_data, hard_data]):
        print(f"Curriculum Learning Stage {stage + 1}")
        dataset = NERDataset(data, word_to_ix, tag_to_ix)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        train_and_evaluate_model_with_mixed_precision(model, data_loader, optimizer, epochs=int(epochs / 3))


