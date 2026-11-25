from __future__ import annotations

import unittest

from kefir.tokenizer import EOD_TOKEN_ID, KefirTokenizer, load_tokenizer


class KefirTokenizerTests(unittest.TestCase):
    def test_text_roundtrip_does_not_promote_literal_controls(self) -> None:
        codec = load_tokenizer()
        text = "Hello <think> e già."
        encoded = codec.encode(text)
        self.assertTrue(encoded)
        self.assertTrue(all(0 <= token_id < EOD_TOKEN_ID for token_id in encoded))
        self.assertEqual(codec.decode(encoded), text)

    def test_wrong_asset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            KefirTokenizer(b"{}")


if __name__ == "__main__":
    unittest.main()
