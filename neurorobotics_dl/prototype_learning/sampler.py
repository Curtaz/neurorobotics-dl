from typing import Any, Dict, Iterator, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


class EpisodicSampler(object):
    """Implement an Episodic sampler."""

    def __init__(self,
                 data: Sequence[Tuple[Tuple[Tensor,Tensor], Tensor]],
                 n_support: int,
                 n_query: int,
                 n_episodes: int,
                 n_classes: int = None,
                 balance_query: bool = False) -> None:
        """Initialize the EpisodicSampler.

        Parameters
        ----------
        data: Sequence[Tuple[torch.Tensor, torch.Tensor]]
            The input data as a list of (sequence, label) pairs
        n_support : int
            The number of support points per class
        n_query : int
            If balance_query is True, this should be the number
            of query points per class, otherwise, this is the total
            number of query points for the episode
        n_episodes : int
            Number of episodes to run in one "epoch"
        n_classes : int, optional
            The number of classes to sample per episode, defaults to all
        pad_index : int, optional
            The padding index used on sequences.
        balance_query : bool, optional
            If True, the same number of query points are sampled per
            class, otherwise query points are sampled uniformly
            from the input data.

        """

        self.n_support = n_support
        self.n_query = n_query
        self.n_classes = n_classes
        self.n_episodes = n_episodes

        self.balance_query = balance_query

        if len(data) == 0:
            raise ValueError("No examples provided")

        # Split dataset by target
        self.target_to_examples: Dict[int, Any] = dict()
        for source, target in data:
            self.target_to_examples.setdefault(int(target), []).append((source, target))

        self.all_classes = list(self.target_to_examples.keys())

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Sample from the list of features and yields batches.

        Yields
        ------
        Iterator[Tuple[Tensor, Tensor, Tensor, Tensor]]
            In order: the query_source, the query_target
            the support_source, and the support_target tensors.
            For sequences, the batch is used as first dimension.

        """
        for _ in range(self.n_episodes):
            # Sample n_classes to run a training episode over
            classes = self.all_classes
            if self.n_classes is not None:
                classes = list(np.random.permutation(self.all_classes))[:self.n_classes]

            # Sample n_support and n_query points per class
            supports, queries = [], []
            for i, target_class in enumerate(classes):
                examples = self.target_to_examples[target_class]
                indices = np.random.permutation(len(examples))
                supports.extend([(examples[j][0], i) for j in indices[:self.n_support]])

                if self.balance_query:
                    query_indices = indices[self.n_support:self.n_support + self.n_query]
                    queries.extend([(examples[j][0], i) for j in query_indices])
                else:
                    queries.extend([(examples[j][0], i) for j in indices[self.n_support:]])

            if not self.balance_query:
                indices = np.random.permutation(len(queries))
                queries = [queries[i] for i in indices[:self.n_query]]

            query_source, query_target = list(zip(*queries))
            support_source, support_target = list(zip(*supports))

            query_source = torch.from_numpy(np.array(query_source,dtype=np.float32))
            query_target = torch.tensor(query_target)

            support_source = torch.from_numpy(np.array(support_source,dtype=np.float32))
            support_target = torch.tensor(support_target)

            if len(query_target.size()) == 2:
                query_target = query_target.squeeze()
            if len(support_target.size()) == 2:
                support_target = support_target.squeeze()

            yield (query_source,
                   query_target.long(),
                   support_source,
                   support_target.long())
    def __len__(self):
      return self.n_episodes



class BaseSampler(object):

    def __init__(self,
                 data: Sequence,
                 shuffle: bool = True,
                 batch_size: int = 16):
        """A basic sampler.

        Parameters
        ----------
        data : Sequence[Tuple[Tensor, Tensor]]
            The input data to sample from, as a list of
            (source, target) pairs
        shuffle : bool, optional
            Whether to shuffle the data, by default True
        batch_size : int, optional
            The batch size to use, by default 16

        """
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        """Sample from the list of features and yields batches.

        Yields
        ------
        Iterator[Tuple[Tensor, Tensor]]
            In order: source, target
            For sequences, the batch is used as first dimension.

        """
        if self.shuffle:
            indices = np.random.permutation(len(self.data))
        else:
            indices = list(range(len(self.data)))

        num_batches = len(indices) // self.batch_size
        indices_splits = np.array_split(indices, num_batches)
        for split in indices_splits:
            examples = [self.data[i] for i in split]
            source, target = list(zip(*examples))
            source = torch.from_numpy(np.array(source,dtype=np.float32))
            target = torch.tensor(target)
            yield (source, target.long())

    def __len__(self):
      return len(self.data)//self.batch_size
      

