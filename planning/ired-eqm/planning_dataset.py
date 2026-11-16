import random
import numpy as np
import gzip
import pickle
from tqdm.auto import tqdm
from reasoning_dataset import random_generate_graph, random_generate_graph_dnc, random_generate_special_graph
from torch.utils.data.dataset import Dataset


class PlanningDataset(Dataset):
    def __init__(self, dataset_identifier: str, split: str, num_identifier='100000'):
        self._dataset_identifier = dataset_identifier
        self._split = split
        self._num_identifier = num_identifier

        if self._dataset_identifier in ('shortest-path-15-1d', 'shortest-path-25-1d'):
            self._init_load_1d()
        else:
            raise ValueError('Unknown dataset identifier: {}.'.format(self._dataset_identifier))

        self.inp_dim = self._all_condition[0].shape[-1]
        self.out_dim = self._all_output[0].shape[-1]

    @classmethod
    def _load_data_raw(cls, identifier, num_identifier):
        if hasattr(cls, f'_data_{identifier}_{num_identifier}_raw'):
            return getattr(cls, f'_data_{identifier}_{num_identifier}_raw')

        with gzip.open('data/planning/{}-{}.pkl.gz'.format(identifier, num_identifier), 'rb') as f:
            all_data = pickle.load(f)

        setattr(cls, f'_data_{identifier}_{num_identifier}_raw', all_data)
        return all_data

    def _init_load_1d(self):
        print('Loading dataset {}-{}...'.format(self._dataset_identifier, self._num_identifier))

        if self._dataset_identifier.startswith('shortest-path') and self._dataset_identifier.endswith('1d'):
            pass
        else:
            raise NotImplementedError('1D inputs are only supported for shortest-path.')

        # remove the -1d suffix
        all_data = self._load_data_raw(self._dataset_identifier[:-3], self._num_identifier)
        if self._split == 'train':
            all_data = all_data[:int(0.9 * len(all_data))]
        elif self._split == 'validation':
            all_data = all_data[int(0.9 * len(all_data)):int(1.0 * len(all_data))]
        else:
            raise ValueError('Unknown split: {}.'.format(self._split))

        padding = 8
        all_condition, all_output = list(), list()
        for data in tqdm(all_data, desc='Preprocessing the data'):
            states = data['states']
            actions = data['actions']

            n = states[0].shape[0]
            actions = actions + [actions[-1] for _ in range(padding - len(actions))]
            actions = np.array(actions, dtype='int32')
            actions_onehot = np.zeros((padding, n, 1), dtype=np.float32)
            actions_onehot[np.arange(padding), actions, 0] = 1

            condition = np.concatenate([states[0], states[-1]], axis=-1)
            output = actions_onehot
            all_condition.append(condition)
            all_output.append(output)

            # print(condition[:, 0, 1].argmax(), condition[:, 0, 3].argmax(), actions)

        self._all_condition = np.stack(all_condition, axis=0)
        self._all_output = np.stack(all_output, axis=0)

        # normalize to -1 to 1
        self._all_condition = (self._all_condition - 0.5) * 2
        self._all_output = (self._all_output - 0.5) * 2

        print('Finished loading dataset {}-{}...'.format(self._dataset_identifier, self._num_identifier))

    def __len__(self):
        return self._all_condition.shape[0]

    def __getitem__(self, index):
        return self._all_condition[index], self._all_output[index]


class GraphEnvBase(object):
    """Graph Env Base."""

    def __init__(self, nr_nodes, p=0.5, directed=True, gen_method='edge'):
        """Initialize the graph env.

        Args:
            n: The number of nodes in the graph.
            p: Parameter for random generation. (Default 0.5)
                (edge method): The probability that a edge doesn't exist in directed graph.
                (dnc method): Control the range of the sample of out-degree.
                other methods: Unused.
            directed: Directed or Undirected graph. Default: ``False``(undirected)
            gen_method: Use which method to randomly generate a graph.
                'edge': By sampling the existance of each edge.
                'dnc': Sample out-degree (:math:`m`) of each nodes, and link to nearest neighbors in the unit square.
                'list': generate a chain-like graph.
        """
        super().__init__()
        self._nr_nodes = nr_nodes
        self._p = p
        self._directed = directed
        self._gen_method = gen_method
        self._graph = None

    @property
    def graph(self):
        return self._graph

    def _gen_graph(self):
        """ generate the graph by specified method. """
        n = self._nr_nodes
        p = self._p
        if self._gen_method in ['edge', 'dnc']:
            gen = random_generate_graph if self._gen_method == 'edge' else random_generate_graph_dnc
            self._graph = gen(n, p, self._directed)
        else:
            self._graph = random_generate_special_graph(n, self._gen_method, self._directed)


class GraphPathEnv(GraphEnvBase):
    """Env for Finding a path from starting node to the destination."""

    def __init__(self, nr_nodes, dist_range, p=0.5, directed=True, gen_method='dnc'):
        super().__init__(nr_nodes, p, directed, gen_method)
        self._dist_range = dist_range

    @property
    def dist(self):
        return self._dist

    def reset(self):
        """Restart the environment."""
        self._dist = self._sample_dist()
        self._task = None
        while True:
            self._gen_graph()
            task = self._gen_task()
            if task is not None:
                break
        self._dist_matrix = task[0]
        self._task = (task[1], task[2])
        self._current = self._task[0]
        self._steps = 0
        return self.get_state()

    def _sample_dist(self):
        lower, upper = self._dist_range
        upper = min(upper, self._nr_nodes - 1)
        return random.randint(0, upper - lower + 1) + lower

    def _gen_task(self):
        """Sample the starting node and the destination according to the distance."""
        dist_matrix = self._graph.get_shortest()
        st, ed = np.where(dist_matrix == self.dist)
        if len(st) == 0:
            return None
        ind = random.randint(0, len(st) - 1)
        return dist_matrix, st[ind], ed[ind]

    def get_state(self):
        relation = self._graph.get_edges()
        current_state = np.zeros_like(relation)
        current_state[self._current, :] = 1
        return np.stack([relation, current_state], axis=-1)

    def step(self, action):
        """Move to the target node from current node if has_edge(current -> target)."""
        if self._current == self._task[1]:
            return self.get_state(), 1, True, {}
        if self._graph.has_edge(self._current, action):
            self._current = action
        else:
            return self.get_state(), -1, False, {}
        if self._current == self._task[1]:
            return self.get_state(), 1, True, {}
        self._steps += 1
        if self._steps >= self.dist:
            return self.get_state(), 0, True, {}
        return self.get_state(), 0, False, {}

    def oracle_policy(self, state):
        """Oracle policy: Swap the first two numbers that are not sorted."""
        current = self._current
        target = self._task[1]
        if current == target:
            return target
        possible_actions = state[current, :, 0] == 1
        # table = list()
        # table.append(('connected', possible_actions.nonzero()[0]))
        possible_actions = possible_actions & (self._dist_matrix[:, target] < self._dist_matrix[current, target])
        # table.append(('dist', possible_actions.nonzero()[0]))
        # print(tabulate(table, headers=['name', 'list']))
        if np.sum(possible_actions) == 0:
            raise RuntimeError('No action found.')
        return np.random.choice(np.where(possible_actions)[0])

    def generate_data(self, nr_data_points: int):
        data = list()
        for _ in tqdm(range(nr_data_points)):
            obs = self.reset()
            states, actions = [obs], list()
            while True:
                action = self.oracle_policy(obs)
                if action is None:
                    raise RuntimeError('No action found.')
                obs, reward, finished, _ = self.step(action)
                states.append(obs)
                actions.append(action)

                assert reward >= 0

                if finished:
                    break
            # import ipdb; ipdb.set_trace()
            data.append({'states': states, 'actions': actions, 'optimal_steps': self._dist, 'actual_steps': len(actions)})
        return data

