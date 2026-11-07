import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import localbrain.corpus as corpus


def make_inputs(folder):
    code = [{"code": f"def f{i}(x):\n    # helper number {i}\n    return x * {i} + {i % 7}\n" * 3,
             "language": "Python", "license": "mit"} for i in range(900)]
    english = [{"text": f"Lesson {i}. Water boils at one hundred degrees under standard pressure. " * 6,
                "int_score": 4} for i in range(300)]
    pq.write_table(pa.Table.from_pylist(code), folder / "code.parquet")
    pq.write_table(pa.Table.from_pylist(english), folder / "en.parquet")
    return [folder / "code.parquet"], [folder / "en.parquet"]


class CorpusResumeTests(unittest.TestCase):
    def setUp(self):
        self.saved_batch, self.saved_iter = corpus.BATCH_DOCS, corpus.iter_documents
        corpus.BATCH_DOCS = 50

    def tearDown(self):
        corpus.BATCH_DOCS, corpus.iter_documents = self.saved_batch, self.saved_iter

    def build(self, root, code, en, name, **kw):
        corpus.build_corpus(code, en, root / name, vocab_size=300, max_tokens=10**9,
                            tokenizer_chars=200_000, **kw)
        return {s: (root / name / f"{s}.bin").read_bytes() for s in ("train", "val")}, \
            json.loads((root / name / "meta.json").read_text())

    def test_interrupted_build_resumes_to_identical_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            code, en = make_inputs(root)
            reference, ref_meta = self.build(root, code, en, "reference")

            # Same stream, but the machine "dies" after 700 documents; checkpoints happen every batch.
            original = corpus.iter_documents
            calls = []

            def dies_midway(*a, **k):
                calls.append(1)
                for n, item in enumerate(original(*a, **k)):
                    if len(calls) == 2 and n == 725:  # call 1 trains the tokenizer, call 2 encodes
                        raise RuntimeError("machine lost")
                    yield item
            corpus.iter_documents = dies_midway
            ckpt = root / "ckpt"
            with self.assertRaises(RuntimeError):
                corpus.build_corpus(code, en, root / "resumed", vocab_size=300, max_tokens=10**9,
                                    tokenizer_chars=200_000, checkpoint=ckpt, checkpoint_minutes=0)
            progress = json.loads((ckpt / "progress.json").read_text())
            self.assertGreater(progress["docs"], 0)
            self.assertEqual(progress["docs"], 700)  # last checkpoint before the crash at 725
            # Simulate a partial write past the recorded progress; resume must cut it off.
            with (ckpt / "train.bin").open("ab") as f:
                f.write(b"\x01\x02junk")
            self.assertFalse((root / "resumed").exists())

            corpus.iter_documents = original
            resumed, meta = self.build(root, code, en, "resumed", checkpoint=ckpt, checkpoint_minutes=0)
            self.assertEqual(resumed, reference)
            self.assertEqual(meta["tokens"], ref_meta["tokens"])
            self.assertEqual(meta["tokenizer_sha256"], ref_meta["tokenizer_sha256"])
            self.assertEqual(meta["mixture_tokens"], ref_meta["mixture_tokens"])

    def test_checkpoint_rejects_different_settings(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            code, en = make_inputs(root)
            ckpt = root / "ckpt"
            self.build(root, code, en, "a", checkpoint=ckpt, checkpoint_minutes=0)
            with self.assertRaises(ValueError):
                corpus.build_corpus(code, en, root / "b", vocab_size=300, max_tokens=5000,
                                    tokenizer_chars=200_000, checkpoint=ckpt)


if __name__ == "__main__":
    unittest.main()
