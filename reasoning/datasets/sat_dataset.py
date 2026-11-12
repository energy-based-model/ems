from torch.utils.data import Dataset
import torch
import cnfgen
import os

class SATDataset(Dataset):
    def __init__(self, path):
        solutions = []
        negations = []

        for file in os.listdir(path):
            if not file.endswith('.cnf'):
                continue

            cnf = cnfgen.utils.parsedimacs.from_dimacs_file(
                cnfgen.formula.cnf.CNF,
                fileorname=os.path.join(path, file),
            )
            clauses = torch.tensor(cnf.clauses(), dtype=torch.long)

            sol_path = os.path.join(path, file.replace('.cnf', '.sol'))
            with open(sol_path) as f:
                sol = torch.tensor(list(map(int, f.read().split())))

            indices = clauses.abs() - 1
            solutions.append((sol[indices] > 0).float())
            negations.append((clauses < 0).float())

        self.solutions = torch.cat(solutions, dim=0)
        self.negation = torch.cat(negations, dim=0)

    def __len__(self):
        return self.negation.shape[0]

    def __getitem__(self, index):
        # The contrastive negative makes every literal false: positive literals
        # take value 0 and negated literals take value 1, i.e. the negation mask.
        return self.solutions[index], self.negation[index], self.negation[index]


class SATEvalDataset(Dataset):
    def __init__(self, path):
        self.instances = []
        files = [f for f in os.listdir(path) if f.endswith('.cnf')]
        
        for file in files:
            cnf = cnfgen.utils.parsedimacs.from_dimacs_file(
                cnfgen.formula.cnf.CNF,
                fileorname=os.path.join(path, file)
            )
            clauses = torch.tensor(cnf.clauses(), dtype=torch.long)
            indices = torch.abs(clauses) - 1
            negation = (clauses < 0).float()
            num_variables = cnf.number_of_variables()
            self.instances.append((indices, negation, num_variables))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]
