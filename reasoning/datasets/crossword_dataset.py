from torch.utils.data import Dataset
import torch
import pickle
import json

class CrosswordDataset(Dataset):
    def __init__(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)

        self.embeddings = torch.tensor([emb for _, _, emb in data], dtype=torch.float32)
        eye = torch.eye(26)
        indices = torch.tensor([[ord(c) - ord('A') for c in word] for _, word, _ in data])
        self.words = eye[indices].flatten(1)
    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, index):
        return self.words[index], self.embeddings[index]
    
class CrosswordEvalDataset(Dataset):
    def __init__(self, test_path, json_path):
        self.clue_embeddings_dict = load_clue_embeddings_dict(test_path)
        self.instances = load_instances(json_path)
        self.eye = torch.eye(26)

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index):
        words, h_clues, v_clues = self.instances[index]

        h_clues_embeddings = torch.tensor(
            [self.clue_embeddings_dict[clue] for clue in h_clues], 
            dtype=torch.float32
        )
        v_clues_embeddings = torch.tensor(
            [self.clue_embeddings_dict[clue] for clue in v_clues], 
            dtype=torch.float32
        )

        indices = torch.tensor([ord(c) - ord('A') for c in words])
        target = self.eye[indices]

        return target, h_clues_embeddings, v_clues_embeddings

def load_clue_embeddings_dict(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)

    clue_embeddings_dict = {}

    for clue, _, embedding in data:
        clue_embeddings_dict[clue] = embedding

    return clue_embeddings_dict

def load_instances(json_path, split='test'):
    with open(json_path) as file:
        data = json.load(file)
    
    if split == 'test':
        data = data[1:100:5]
    elif split == 'train':
        test_set = set(range(1, 100, 5))
        data = [item for i, item in enumerate(data) if i not in test_set]
    else:
        raise ValueError('Split must be train or test.')
    
    instances = []

    for datum in data:
        clues = datum[0]
        h_clues = clues[:5]
        v_clues = clues[5:]

        words = datum[1]

        instances.append((words, h_clues, v_clues))

    return instances
