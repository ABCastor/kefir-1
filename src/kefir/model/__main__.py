"""Explicit MoE process entrypoint; shares the canonical training loop."""
from __future__ import annotations

from kefir.trainer.cli import main as train
from kefir.trainer.cli_args import parse_args as parse_trainer_args
from kefir.trainer.cli_args import parser as trainer_parser

from .compile_lane import COMPILE_MODES, DEFAULT_COMPILE_MODE
from .setup import setup, validation_reader


def _enable_accepted_model_size(parser) -> None:
    """Select the one supported public architecture."""

    model_size_action = next(
        action for action in parser._actions if getattr(action, "dest", None) == "model_size"
    )
    model_size_action.choices = ("accepted",)


def parse_args(argv=None):
    parser = trainer_parser()
    _enable_accepted_model_size(parser)
    parser.add_argument(
        "--load-balancing",
        choices=("quantile",),
        default=None,
        help=(
            "Confirm the accepted Quantile Balancing controller."
        ),
    )
    parser.add_argument("--balancing-step-size", type=float, default=0.0)
    parser.add_argument(
        "--compile",
        dest="compile_mode",
        choices=COMPILE_MODES,
        default=DEFAULT_COMPILE_MODE,
        help=(
            "Execution lane only, never checkpoint-visible architecture. "
            "'blocks' wraps every MoE decoder block in torch.compile after the "
            "model is built; state-dict keys and numerics of the Quantile "
            "Balancing bias are unchanged. Default is off."
        ),
    )
    args = parse_trainer_args(argv, argument_parser=parser)
    if args.balancing_step_size != 0.0:
        parser.error("accepted Quantile Balancing does not use --balancing-step-size")
    return args


def main(argv=None):
    from kefir.trainer.distributed import shutdown_distributed
    try:
        return train(argv, setup_fn=setup, validation_reader_fn=validation_reader,
                     parse_args_fn=parse_args)
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
