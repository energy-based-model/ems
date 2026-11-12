
from torch.utils.data import Dataset
import torch

class NQueensDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = []

        with open(path, 'r') as file:
            solutions = [line.strip() for line in file if line.strip()]

        self.N = int(len(solutions[0])**0.5)

        for solution in solutions:
            board = torch.tensor([1.0 if c == 'Q' else 0.0 for c in solution]).reshape(self.N, self.N)
            for r in range(self.N):
                self.data.append(board[r])
                self.data.append(board[:, r])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        positive = self.data[index]

        choice = torch.randint(0, 3, (1,)).item()

        if choice == 0:
            negative = positive.clone()
            zeros = (negative == 0).nonzero(as_tuple=True)[0]
            idx = zeros[torch.randint(len(zeros), (1,))]
            negative[idx] += torch.rand(1).item() * 0.5 + 0.5
        elif choice == 1:
            negative = positive.clone()
            ones = (negative == 1).nonzero(as_tuple=True)[0]
            negative[ones] -= torch.rand(ones.shape[0]) * 0.5 + 0.5
        else:
            negative = torch.zeros_like(positive)

        return positive, negative