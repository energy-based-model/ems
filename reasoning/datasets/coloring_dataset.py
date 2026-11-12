import torch
from torch.utils.data import Dataset

class ColoringDataset(Dataset):
    def __init__(self, num_colors):
        self.num_colors = num_colors
        pairs = torch.cartesian_prod(torch.arange(num_colors), torch.arange(num_colors))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        eye = torch.eye(num_colors)
        left_nodes = eye[pairs[:, 0]]
        right_nodes = eye[pairs[:, 1]]
        self.nodes = torch.cat((left_nodes, right_nodes), dim=1)
    
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, index):
        positive = self.nodes[index]

        i = positive[:self.num_colors].argmax()
        j = positive[self.num_colors:].argmax()

        neg1 = torch.zeros_like(positive)
        rand_index = i if torch.rand(1).item() < 0.5 else j
        neg1[rand_index] = torch.empty(1).uniform_(0.5, 1.0).item()
        neg1[rand_index + self.num_colors] = torch.empty(1).uniform_(0.5, 1.0).item()

        neg2 = positive.clone()
        zero_indices = (positive == 0).nonzero(as_tuple=True)[0]
        rand_index = zero_indices[torch.randint(0, len(zero_indices), (1,))].item()
        neg2[rand_index] = torch.empty(1).uniform_(0.01, 1.0).item()

        negatives = torch.stack((neg1, neg2), dim=0)

        return positive, negatives
    
def parse_graph_file(filepath):
    with open(filepath) as f:
        lines = [line.strip() for line in f if line.strip()]

    edges = []
    num_nodes = None
    chrom_number = None
    i = 0

    while i < len(lines):
        if lines[i].startswith("DIMENSION"):
            num_nodes = int(lines[i].split(":")[1])
        elif lines[i] == "EDGE_DATA_SECTION":
            i += 1
            while lines[i] != "-1":
                u, v = map(int, lines[i].split())
                edges.append((u, v))
                i += 1
        elif lines[i] == "CHROM_NUMBER": 
            i += 1
            chrom_number = int(lines[i])
        i += 1

    edges = torch.tensor(edges, dtype=torch.long)

    u = torch.minimum(edges[:, 0], edges[:, 1])
    v = torch.maximum(edges[:, 0], edges[:, 1])

    edges = torch.stack([u, v], dim=1)
    edges = torch.unique(edges, dim=0)

    return num_nodes, edges, chrom_number