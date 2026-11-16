# Unifying Reasoning and Planning through Energy Minimization

Code for *Unifying Reasoning and Planning through Energy Minimization* (NeurIPS 2026).

[Project website](https://energy-based-model.github.io/ems/)

## Setup

The code was run with Python 3.10. Each part has its own requirements:

```bash
pip install -r reasoning/requirements.txt
pip install -r planning/ired-eqm/requirements.txt
```

## Reasoning

Run every command from the `reasoning/` directory:

```bash
cd reasoning
```

### Data

All reasoning data come from [Oarga & Du (2025)](https://github.com/alexOarga/compositional_reasoning). Clone their repository first:

```bash
git clone https://github.com/alexOarga/compositional_reasoning.git
```

**Graph coloring and N-Queens** are in the repository:

```bash
mkdir -p data/coloring
cp compositional_reasoning/comp_reasoning/data/random_graphs/*.graph data/coloring/
cp compositional_reasoning/comp_reasoning/data/nqueens/*.txt data/nqueens/
```

**3-SAT:** download the data from [this link](https://drive.google.com/file/d/1FGPS0Ox4lOp5UCaFRqwfVTy2SvtqOAuQ/view?usp=sharing) (from their README), and put its `train/`, `valid/`, and `test/` folders in `data/sat/`. The test formulas should end up in `data/sat/test/uf20-91/`.

**Crosswords:** download the precomputed clue embeddings from [this link](https://drive.google.com/file/d/1_YteXjmiRs4T4GhBJgHCZaL7y279IRvI/view?usp=sharing) (from their README), and put `train.pkl`, `valid.pkl`, and `test.pkl` in `data/crossword/`. Then copy the puzzles:

```bash
cp compositional_reasoning/comp_reasoning/data/crosswords_qa/mini0505.json data/crossword/
```

The finished layout:

```
data/
├── coloring/    *.graph
├── nqueens/     {N}_queens_{train,valid,test}.txt
├── sat/         train/  valid/  test/uf20-91/
└── crossword/   train.pkl  valid.pkl  test.pkl  mini0505.json
```

### Training and Evaluation

Each task has a `train_<task>.py` and an `eval_<task>.py`. Checkpoints are saved to `checkpoints/`.

```bash
# Graph coloring
python train_coloring.py
python eval_coloring.py

# N-Queens
python train_nqueens.py
python eval_nqueens.py

# 3-SAT (seeds 0, 13, 42)
python train_sat.py --seed 0
python eval_sat.py --seed 0

# Crosswords
python train_crossword.py
python eval_crossword.py
```

### Settings

All training and search settings for a task (model size, λ, β, epochs, search method, candidates, refinement steps, step size, data paths) live in `configs/<task>_config.py`. The defaults are the settings we used to reproduce the paper's results. To change a setting, edit the config file or override the field from the command line:

```bash
python eval_nqueens.py --n_queen 9 --num_particles 1024
python eval_sat.py --seed 13
```

Results can differ slightly depending on the GPU, and the best target scale (`--lambd`) can shift with it. If a run falls a little short of the paper, a few more refinement steps (`--sampling_steps`) or candidates (`--num_particles`) usually close the gap. For example, our retrained 8-Queens model (λ = 12) solves all 100 boards with 256 candidates, a quarter of DiffEBM's 1,024.

## Planning

### Shortest Path

Run every command from the `planning/ired-eqm/` directory:

```bash
cd planning/ired-eqm
```

#### Data

The graphs are generated locally with IRED's generator (15-node graphs for training, 25-node graphs for generalization):

```bash
python gen_planning_dataset.py shortest-path-15 --size 100000
python gen_planning_dataset.py shortest-path-25 --size 10000
```

#### Training and Evaluation

```bash
python train.py
python infer.py --ckpt results/ds_shortest-path-15-1d/model_gnn-conv-1d-v2_eqm_lam1.0_bs512/model-90.pt
```

The defaults are the paper's settings. Training evaluates on both graph sizes as IRED does and saves a checkpoint every 1,000 updates; the paper uses `model-90.pt`. `first_action_accuracy_2` is the reported metric.

`infer.py` seeds the starting noise (`--seed`, default 0), so repeated runs on the same machine give the same numbers. To try other starting noise:

```bash
python infer.py --ckpt <checkpoint> --seed 1
```

Results can still differ from the paper by about a point. The training-time evaluation does not seed its starting noise, and each evaluation scores 1,408 graphs per graph size, so different starting noise changes which minimum some graphs end in. The graph generator is not seeded either, so each generated dataset has different graphs, and results can also vary with the GPU.

## Acknowledgements

The reasoning task data, encodings, and evaluation protocols follow [Oarga & Du (2025)](https://arxiv.org/abs/2510.20607). The crossword puzzles are from Tree of Thoughts (Yao et al., 2023). The 3-SAT test formulas are from SATLIB.

The shortest-path code builds on [IRED](https://github.com/yilundu/ired_code_release) (Du et al., 2024), including its graph generator, backbone, negatives, training loop, and evaluation metrics. IRED's training loop derives from [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) by Phil Wang, released under the MIT License (`planning/ired-eqm/LICENSE`).

## Citation

```bibtex
@inproceedings{rodriguez2026ems,
  title     = {Unifying Reasoning and Planning through Energy Minimization},
  author    = {Rodriguez, Adrian and Kim, Angelica and Guo, Yunhui and Du, Yilun},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
