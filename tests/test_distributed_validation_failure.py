"""Both ranks must leave validation when rank zero fails."""
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, root):
    from kefir.trainer.cli import main
    from kefir.trainer.distributed import DistributedContext
    dist.init_process_group('gloo', init_method=f'file://{root}/group', rank=rank,
                            world_size=2, timeout=timedelta(seconds=10))
    try:
        engine = SimpleNamespace(distributed=DistributedContext(rank, rank, 2),
                                 device=torch.device('cpu'), global_step=0,
                                 consumed_tokens=0)
        def evaluate(*args, **kwargs):
            raise torch.OutOfMemoryError('synthetic evaluation OOM')
        engine.evaluate = evaluate
        class Session:
            def step(self):
                engine.global_step += 1
                engine.consumed_tokens += 4
                return SimpleNamespace(as_dict=lambda: {'step': engine.global_step, 'loss': 1.})
        reader = SimpleNamespace(iter_from_start=lambda n: iter([object()]))
        setup = lambda args: (object(), SimpleNamespace(evaluation_every_steps=1,
                             checkpoint_every_steps=0), engine, Session(), object())
        with patch('kefir.trainer.cli.configure_remote_publication', return_value=None), \
             patch('kefir.trainer.cli.configure_wandb', return_value=None), \
             patch('kefir.trainer.cli.validation_reader', return_value=reader):
            try:
                main(['--dataset-dir', root, '--checkpoint-dir', root, '--steps', '1',
                      '--validation-blocks', '1'], setup_fn=setup)
            except Exception as error:
                message = f'{type(error).__name__}: {error}'
            else:
                message = 'no error'
        (Path(root)/f'rank{rank}.txt').write_text(message)
    finally:
        dist.destroy_process_group()


def test_evaluation_failure_propagates_to_both_ranks(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert 'synthetic evaluation OOM' in (tmp_path/'rank0.txt').read_text()
    assert 'distributed validation failed on another rank' in (tmp_path/'rank1.txt').read_text()
