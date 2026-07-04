"""Concatenate V3 datasets with integer oversampling for real+pseudo mixing."""


class ConcatV3:
    def __init__(self, datasets, weights=None):
        self.datasets = list(datasets)
        self.weights = list(weights) if weights else [1] * len(self.datasets)
        assert len(self.weights) == len(self.datasets)
        self.heatmap_size = self.datasets[0].heatmap_size
        self._index = []  # (ds_idx, local_idx)
        for di, (d, w) in enumerate(zip(self.datasets, self.weights)):
            for _ in range(int(w)):
                self._index.extend((di, li) for li in range(len(d)))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        di, li = self._index[int(i)]
        return self.datasets[di][li]
