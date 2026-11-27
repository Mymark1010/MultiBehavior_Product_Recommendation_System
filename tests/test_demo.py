import pytest

from rr.demo import prefix_frame


def test_prefix_frame_groups_by_timestamp():
    f, t = prefix_frame([(5, "view", 100), (6, "addtocart", 200), (7, "view", 200), (5, "view", 350)])
    assert t == 350
    assert f["group_index"].tolist() == [1, 2, 2, 3]
    assert f["distance"].tolist() == [2, 1, 1, 0]


def test_prefix_frame_rejects_bad_input():
    with pytest.raises(ValueError):
        prefix_frame([])
    with pytest.raises(ValueError):
        prefix_frame([(5, "click", 1)])
