"""Run the pretraining corpus recipe checks (data/pretraining/test_recipe.py) inside the main suite."""
import sys
import unittest
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parents[1] / "data" / "pretraining"


def test_corpus_recipe_checks_pass():
    sys.path.insert(0, str(CORPUS))
    try:
        suite = unittest.defaultTestLoader.loadTestsFromName("test_recipe")
        result = unittest.TextTestRunner(verbosity=0).run(suite)
    finally:
        sys.path.remove(str(CORPUS))
    assert result.wasSuccessful() and result.testsRun >= 4


def test_missing_student_asset_explains_how_to_obtain_it(tmp_path):
    sys.path.insert(0, str(CORPUS))
    try:
        from student_gate import STUDENT_SHA256, StudentGate
        with pytest.raises(FileNotFoundError, match="obtain students.pkl separately"):
            StudentGate(tmp_path / "students.pkl")
    finally:
        sys.path.remove(str(CORPUS))
    assert len(STUDENT_SHA256) == 64
