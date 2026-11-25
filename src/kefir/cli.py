"""The one Kefir command line: ``kefir <command> ...``.

pretrain   train or resume the MoE language model on schema-v2 shards
sft        render conversations into token rows; fine-tune on them
evaluate   ``text``: held-out NLL and thematic panels; ``chat``: frozen SFT suite
generate   continue raw text with base weights
chat       talk to SFT weights in the terminal
export     convert a trusted trainer state into .safetensors weights
bench      ``build``: reconstruct the frozen chat suite from pinned public sources
"""
from __future__ import annotations

import sys


def _bench(argv):
    import argparse
    import json
    p = argparse.ArgumentParser(prog="kefir bench build",
                                description="Reconstruct the frozen kefir-bench/1 suite")
    p.add_argument("command", choices=("build",))
    p.add_argument("--cache", required=True, help="directory for the pinned upstream files")
    p.add_argument("--output", required=True, help="new directory for the suite")
    p.add_argument("--download", action="store_true", help="fetch missing pinned upstream files")
    a = p.parse_args(argv)
    from kefir.bench.build import build_suite
    print(json.dumps(build_suite(a.output, a.cache, download=a.download)["files"]))
    return 0


def _evaluate(argv):
    if argv[:1] == ["text"]:
        from kefir.evaluate import main
        return main(argv[1:])
    if argv[:1] == ["chat"]:
        from kefir.sft.cli import evaluate_chat_main
        return evaluate_chat_main(argv[1:])
    print("usage: kefir evaluate {text,chat} ...", file=sys.stderr)
    return 2


def _pretrain(argv):
    from kefir.train import main
    return main(argv)


def _sft(argv):
    from kefir.sft.cli import main
    return main(argv)


def _generate(argv):
    from kefir.generate import main
    return main(argv)


def _chat(argv):
    from kefir.sft.chat import main
    return main(argv)


def _export(argv):
    from kefir.export import main
    return main(argv)


COMMANDS = {"pretrain": _pretrain, "sft": _sft, "evaluate": _evaluate, "generate": _generate,
            "chat": _chat, "export": _export, "bench": _bench}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__.strip(), file=sys.stderr)
        return 0 if argv[:1] in (["-h"], ["--help"]) else 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
