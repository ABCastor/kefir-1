import torch

from kefir.trainer.step import _microbatch_to_device, _ordered_batch_tensors
from kefir.trainer.types import IGNORE_INDEX, TokenBatch


def test_masked_routing_positions_do_not_depend_on_microbatch_size():
    inputs = torch.arange(32).reshape(4, 8)
    labels = inputs.clone()
    labels[0, 2:] = IGNORE_INDEX
    labels[1, 4:] = IGNORE_INDEX
    labels[2, 6:] = IGNORE_INDEX
    batch = TokenBatch(0, "train", inputs, labels, 4, int(labels.ne(IGNORE_INDEX).sum()))
    ordered_inputs, ordered_labels = _ordered_batch_tensors(batch)
    counts = []
    for size in (1, 2, 4):
        positions = 0
        for start in range(0, 4, size):
            micro_inputs, _ = _microbatch_to_device(
                ordered_inputs, ordered_labels, start=start,
                stop=min(start + size, 4), device=torch.device("cpu"),
            )
            positions += micro_inputs.numel()
        counts.append(positions)
    assert counts == [32, 32, 32]
