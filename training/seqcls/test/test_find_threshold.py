import pytest
from training.src.find_threshold import get_optimized_threshold


@pytest.mark.parametrize("input, expected", [
    (
        (
            [
                {"precision": 0.8041, "recall": 0.3527, "f1": 0.6562, "threshold": 0.8846},
                {"precision": 0.7211, "recall": 0.3944, "f1": 0.6345, "threshold": 0.7052},
                {"precision": 0.6649, "recall": 0.4015, "f1": 0.5694, "threshold": 0.6781},
                {"precision": 0.5491, "recall": 0.4633, "f1": 0.5151, "threshold": 0.6038},
                {"precision": 0.5188, "recall": 0.5067, "f1": 0.5392, "threshold": 0.5469},
                {"precision": 0.4582, "recall": 0.5346, "f1": 0.4864, "threshold": 0.4857},
                {"precision": 0.4377, "recall": 0.5781, "f1": 0.5141, "threshold": 0.4299},
                {"precision": 0.3958, "recall": 0.6472, "f1": 0.5344, "threshold": 0.3258},
            ],
            "precision",
            "recall",
            0.5,
        ),
        {"precision": 0.5188, "recall": 0.5067, "f1": 0.5392, "threshold": 0.5469},
    ),
    (
        (
            [
                {"precision": 0.8041, "recall": 0.3527, "f1": 0.6562, "threshold": 0.8846},
                {"precision": 0.7211, "recall": 0.3944, "f1": 0.6345, "threshold": 0.7052},
                {"precision": 0.6649, "recall": 0.4015, "f1": 0.5694, "threshold": 0.6781},
                {"precision": 0.5491, "recall": 0.4633, "f1": 0.5151, "threshold": 0.6038},
                {"precision": 0.5188, "recall": 0.5067, "f1": 0.5392, "threshold": 0.5469},
                {"precision": 0.4582, "recall": 0.5346, "f1": 0.4864, "threshold": 0.4857},
                {"precision": 0.4377, "recall": 0.5781, "f1": 0.5141, "threshold": 0.4299},
                {"precision": 0.3958, "recall": 0.6472, "f1": 0.5344, "threshold": 0.3258},
            ],
            "recall",
            "precision",
            0.6,
        ),
        {"precision": 0.6649, "recall": 0.4015, "f1": 0.5694, "threshold": 0.6781},
    ),
])
def test_get_optimized_threshold(input, expected):
    actual = get_optimized_threshold(*input)
    assert actual["threshold"] == expected["threshold"]
