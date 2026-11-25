"""Rank program for the two-rank SFT test (launched by torch.distributed.run)."""
import sys
from pathlib import Path

from kefir.sft.recipe import Execution, Recipe
from kefir.sft.train import train

recipe_path, output = Path(sys.argv[1]), sys.argv[2]
recipe, base = Recipe.from_json(recipe_path)
train(recipe, base, Execution(output=output, device="cpu", threads=1))
