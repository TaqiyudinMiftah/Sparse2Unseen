from sparse2unseen.data.manifest import Sample
from sparse2unseen.data.splits import make_random_split


def samples(n=100):
    return [Sample(str(i), f"{i}.jpg", None, None, float(i)) for i in range(n)]


def test_split_is_deterministic_and_disjoint():
    a = make_random_split(samples(), 0.10, seed=1)
    b = make_random_split(samples(), 0.10, seed=1)
    assert a == b
    assert len(a["labeled_ids"]) == 10
    assert set(a["labeled_ids"]).isdisjoint(a["unlabeled_ids"])
    assert len(a["labeled_ids"]) + len(a["unlabeled_ids"]) == 100
