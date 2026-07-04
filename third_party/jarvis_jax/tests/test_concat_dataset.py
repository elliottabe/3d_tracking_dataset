from jarvis_jax.data.concat import ConcatV3


class Fake:
    heatmap_size = 224

    def __init__(self, n, tag):
        self.n = n
        self.tag = tag

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return (self.tag, i)


def test_concat_len_and_oversample():
    a = Fake(3, "real")
    b = Fake(2, "pseudo")
    ds = ConcatV3([a, b], weights=[1, 2])
    assert len(ds) == 3 * 1 + 2 * 2  # 7
    tags = [ds[i][0] for i in range(len(ds))]
    assert tags.count("real") == 3 and tags.count("pseudo") == 4
    assert ds.heatmap_size == 224
