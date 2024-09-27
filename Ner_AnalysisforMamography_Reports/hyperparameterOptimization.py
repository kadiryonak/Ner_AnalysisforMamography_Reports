
# Save Model and Ner Analysis
def save_model(model, path="ner_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model kaydedildi: {path}")

def ner_analysis(model, report, word_to_ix, tag_to_ix):
    model.eval()
    words = report.split()
    word_idxs = [word_to_ix.get(word, word_to_ix["<PAD>"]) for word in words]
    sentence_tensor = torch.tensor([word_idxs], dtype=torch.long)

    with torch.no_grad():
        outputs = model(sentence_tensor)
        preds = torch.argmax(outputs, dim=2).squeeze(0).tolist()

    ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    tags = [ix_to_tag[pred] for pred in preds]
    return list(zip(words, tags))

