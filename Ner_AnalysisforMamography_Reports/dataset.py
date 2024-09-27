import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Creating Dataset and DataLoader
def collate_fn(batch):
    sentences, tags = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=word_to_ix["<PAD>"])
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=tag_to_ix["<PAD>"])
    return sentences_padded, tags_padded

class NERDataset(Dataset):
    def _init_(self, bio_data, word_to_ix, tag_to_ix):
        self.data = bio_data
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx):
        report, tags = self.data[idx]
        word_idxs = [self.word_to_ix[word] for word in report.split()]
        tag_idxs = [self.tag_to_ix[tag] for tag in tags]
        return torch.tensor(word_idxs, dtype=torch.long), torch.tensor(tag_idxs, dtype=torch.long)


