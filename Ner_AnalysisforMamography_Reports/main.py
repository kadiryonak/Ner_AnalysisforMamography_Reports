from data_preprocessing import load_data, convert_to_bio, build_vocab, build_tag_vocab
from dataset import NERDataset, collate_fn
from model import NERModel
from training import train_model_with_curriculum_learning, train_and_evaluate_model_with_mixed_precision
from analysis_and_save import save_model, ner_analysis

import json
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score

def main():
  
    file_paths = [
       "dataset_path"
    ]
    
    reports, labels = load_data(file_paths)
    bio_data = convert_to_bio(reports, labels)
    
    sentences = [report for report, _ in bio_data]
    tags = [tags for _, tags in bio_data]
    
    word_to_ix = build_vocab(sentences)
    tag_to_ix = build_tag_vocab(tags)
    
    
    word_to_ix["<PAD>"] = len(word_to_ix)
    tag_to_ix["<PAD>"] = len(tag_to_ix)
    with open('word_to_ix.json', 'w', encoding='utf-8') as f:
        json.dump(word_to_ix, f)

    with open('tag_to_ix.json', 'w', encoding='utf-8') as f:
        json.dump(tag_to_ix, f)
    
    
    embedding_dim = 128
    hidden_dim = 64
    model = NERModel(len(word_to_ix), len(tag_to_ix), embedding_dim, hidden_dim)
    
    optimizer = optim.Adam([
        {'params': model.embedding.parameters(), 'lr': 1e-3},
        {'params': model.lstm.parameters(), 'lr': 1e-4},
        {'params': model.attention.parameters(), 'lr': 1e-4},
        {'params': model.hidden2tag.parameters(), 'lr': 1e-3}
    ])
    
    # Curriculum Learning with training
    easy_data, medium_data, hard_data = split_data_by_difficulty(bio_data)
    train_model_with_curriculum_learning(model, easy_data, medium_data, hard_data, word_to_ix, tag_to_ix, optimizer, epochs=3150)
    
    # save model
    save_model(model, "ner_model.pth")
    
    test_report = """Your_Report"""
    
    ner_result = ner_analysis(model, test_report, word_to_ix, tag_to_ix)
    print("\nTest Report:")
    print(test_report)
    print("\nNER Analysis Result:")
    for word, tag in ner_result:
        print(f"{word}: {tag}")

    # F1 with Matplotlib
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), f1_scores, label="F1 Score", color='b')
    plt.title("F1 Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.show(block=True)
