import torch
import torch.optim as optim
from data_processing import load_data, convert_to_bio, build_vocab, build_tag_vocab
from model import NERModel
from curriculum_learning import split_data_by_difficulty, train_model_with_curriculum_learning
from training import save_model

if __name__ == "__main__":
    # Veriyi Yükleme
    file_paths = [
        r"C:\Users\Batuhan Koyuncu\Desktop\TEKNOFEST\Karışık\1\all.jsonl",
        r"C:\Users\Batuhan Koyuncu\Desktop\TEKNOFEST\Karışık\2\all.jsonl",
        r"C:\Users\Batuhan Koyuncu\Desktop\TEKNOFEST\Karışık\3\all.jsonl",
        r"C:\Users\Batuhan Koyuncu\Desktop\TEKNOFEST\Karışık\4\all.jsonl"
    ]
    
    reports, labels = load_data(file_paths)
    bio_data = convert_to_bio(reports, labels)
    
    sentences = [report for report, _ in bio_data]
    tags = [tags for _, tags in bio_data]
    
    word_to_ix = build_vocab(sentences)
    tag_to_ix = build_tag_vocab(tags)

    word_to_ix["<PAD>"] = len(word_to_ix)
    tag_to_ix["<PAD>"] = len(tag_to_ix)

    # Modeli Tanımlama ve Optimizasyon
    embedding_dim = 128
    hidden_dim = 64
    model = NERModel(len(word_to_ix), len(tag_to_ix), embedding_dim, hidden_dim)
    
    optimizer = optim.Adam([
        {'params': model.embedding.parameters(), 'lr': 1e-3},
        {'params': model.lstm.parameters(), 'lr': 1e-4},
        {'params': model.attention.parameters(), 'lr': 1e-4},
        {'params': model.hidden2tag.parameters(), 'lr': 1e-3}
    ])
    
    # Curriculum Learning ile Eğitim
    easy_data, medium_data, hard_data = split_data_by_difficulty(bio_data)
    train_model_with_curriculum_learning(model, easy_data, medium_data, hard_data, word_to_ix, tag_to_ix, optimizer, epochs=3150)
    
    # Modeli Kaydet
    save_model(model, "ner_model.pth")
