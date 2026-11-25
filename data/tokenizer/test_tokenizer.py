"""Small checks for the tokenizer recipe: bundled asset, seed rules, and a tiny two-stage fit.

    python -m unittest test_tokenizer          (from this directory)

The tiny fit runs only when `boundlessbpe` is installed. It trains on a few hundred
kilobytes of generated text in seconds and checks that the export has the published format;
it does not reproduce the published vocabulary, which needs the original seed.
"""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import prepare_seed
import verify_tokenizer

HERE = Path(__file__).resolve().parent

try:
    import boundlessbpe  # noqa: F401
    HAVE_TRAINER = True
except ImportError:
    HAVE_TRAINER = False


def synthetic_corpus(path: Path, docs: int = 2000, seed: int = 0) -> None:
    """Generated English-like text: made-up nouns inside frequent function-word phrases."""
    rng = random.Random(seed)
    syllables = ["ka", "lo", "mi", "ren", "sa", "tor", "vel", "dun", "pri", "gal", "fen", "hob"]
    nouns = sorted({"".join(rng.choice(syllables) for _ in range(rng.randint(2, 3))) for _ in range(400)})
    frames = ["the {} of the {}", "a {} in the {}", "{} and the {}", "to the {} on a {}", "it is a {} for the {}"]
    with path.open("w", encoding="utf-8") as f:
        for _ in range(docs):
            text = " ".join(rng.choice(frames).format(rng.choice(nouns), rng.choice(nouns)).capitalize() + "."
                            for _ in range(rng.randint(3, 8)))
            f.write(json.dumps({"text": text}) + "\n")


class BundledTokenizerTests(unittest.TestCase):
    def test_bundled_asset(self) -> None:
        result = verify_tokenizer.verify(verify_tokenizer.find_asset(HERE))
        self.assertEqual(result["sha256"], verify_tokenizer.SHA256)
        self.assertEqual((result["vocab"], result["merges"]), (8000, 7749))


class ManifestTests(unittest.TestCase):
    def test_sources_match_manifest(self) -> None:
        import hashlib
        manifest = json.loads((HERE / "manifest.json").read_text())
        for name, digest in manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(manifest["source_sha256"]["filters.py"], manifest["original_source_sha256"]["filters.py"])


class SeedRuleTests(unittest.TestCase):
    def test_v2_refinement(self) -> None:
        self.assertEqual(prepare_seed.v2_reason("Let f(x, y) be given."), "function_notation")
        self.assertEqual(prepare_seed.v2_reason("Then 3 ** 2 is nine."), "numeric_formula")
        self.assertEqual(prepare_seed.v2_reason("If p -> q holds."), "ascii_symbolic_logic")
        self.assertIsNone(prepare_seed.v2_reason("A plain sentence about the weather."))

    def test_stage_split_with_stub_filter(self) -> None:
        class Accept:
            def assess(self, text):
                return None, {"topics": ["general_prose"]}

        para = " ".join(["word"] * 45)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "val.jsonl").write_text(json.dumps({"text": para}) + "\n")
            rows = [f"Document number {i} talks about the river." for i in range(200)]
            rows += [rows[0], "Let f(x) be odd.", "Intro.\n" + para]
            (tmp / "train.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in rows))
            receipt = prepare_seed.build([tmp / "val.jsonl"], [tmp / "train.jsonl"], tmp / "out", Accept())
            self.assertEqual(receipt["counts"]["reject_exact_duplicate"], 1)
            self.assertEqual(receipt["counts"]["reject_v2_function_notation"], 1)
            self.assertEqual(receipt["counts"]["reject_validation_paragraph_overlap"], 1)
            self.assertEqual(receipt["stage1_docs"], 200)
            stage2 = [json.loads(x)["text"] for x in (tmp / "out/stage2.jsonl").read_text().splitlines()]
            expected = [t for t in rows[:200]
                        if prepare_seed.stage2_selected(prepare_seed.digest(prepare_seed.normalized(t)))]
            self.assertEqual(stage2, expected)


@unittest.skipUnless(HAVE_TRAINER, "boundlessbpe is not installed")
class TinyFitTests(unittest.TestCase):
    def test_tiny_fit_has_published_format(self) -> None:
        import train_tokenizer
        from tokenizers import Tokenizer
        published = json.loads(verify_tokenizer.find_asset(HERE).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            synthetic_corpus(tmp / "stage1.jsonl")
            synthetic_corpus(tmp / "stage2.jsonl", docs=600, seed=1)
            receipt = train_tokenizer.train(tmp / "stage1.jsonl", tmp / "stage2.jsonl", tmp / "tok",
                                            stage1_vocab=400, supermerges=40)
            data = json.loads((tmp / "tok/tokenizer.json").read_text(encoding="utf-8"))
            tok = Tokenizer.from_file(str(tmp / "tok/tokenizer.json"))
        for key in ("version", "truncation", "padding", "normalizer", "pre_tokenizer", "post_processor", "decoder"):
            self.assertEqual(data[key], published[key], key)
        flags = lambda m: {k: v for k, v in m.items() if k not in ("vocab", "merges")}
        self.assertEqual(flags(data["model"]), flags(published["model"]))
        self.assertEqual([t["content"] for t in data["added_tokens"]], verify_tokenizer.SPECIAL)
        self.assertEqual([t["id"] for t in data["added_tokens"]], list(range(440, 448)))
        self.assertEqual(receipt["eod_id"], 440)
        # Same byte alphabet, in the same id order, as the published file.
        inv = {i: t for t, i in data["model"]["vocab"].items()}
        pub = {i: t for t, i in published["model"]["vocab"].items()}
        self.assertEqual([inv[i] for i in range(243)], [pub[i] for i in range(243)])
        # Merges: stage-1 merges then exactly 40 supermerges, as [left, right] string pairs.
        self.assertEqual(len(data["model"]["merges"]), 400 - 243 + 40)
        self.assertTrue(all(isinstance(m, list) and len(m) == 2 for m in data["model"]["merges"]))
        supers = [inv[i] for i in range(400, 440)]
        self.assertTrue(any("Ġ" in t[1:] for t in supers), supers)
        text = "The kalo of the mitor, and it is a renvel for the sa."
        self.assertEqual(tok.decode(tok.encode(text).ids), text)


if __name__ == "__main__":
    unittest.main()
