"""Exercise real image files, the CLI, checkpoint evaluation, and resume."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


class TrainingTests(unittest.TestCase):
    def test_imagefolder_warm_joint_evaluate_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, output = directory / "data", directory / "output"
            for split in ("train", "val"):
                for label, color in (("alpha", (200, 30, 20)), ("beta", (20, 180, 30))):
                    folder = data / split / label
                    folder.mkdir(parents=True)
                    for index in range(2):
                        Image.new("RGB", (80, 96), color).save(folder / f"{index}.png")
            base = [sys.executable, str(ROOT / "train.py"), "--data-root", str(data),
                    "--output", str(output), "--image-size", "64", "--batch-size", "2",
                    "--warm-epochs", "1", "--device", "cpu", "--threads", "2"]

            def run(arguments):
                result = subprocess.run(arguments, cwd=ROOT, text=True, capture_output=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout

            run(base + ["--from-scratch", "--epochs", "2"])
            records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["stage"] for r in records], ["warm", "joint"])
            self.assertTrue((output / "best.pt").is_file())
            last = output / "last.pt"
            saved = torch.load(last, map_location="cpu", weights_only=True)
            self.assertEqual(saved["class_to_idx"], {"alpha": 0, "beta": 1})
            evaluated = json.loads(run(base + ["--eval-only", str(last)]))
            self.assertEqual(evaluated["validation"]["samples"], 4)
            self.assertAlmostEqual(evaluated["validation"]["loss"], records[-1]["validation"]["loss"], places=6)
            run(base + ["--resume", str(last), "--epochs", "3"])
            resumed = torch.load(last, map_location="cpu", weights_only=True)
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(len((output / "metrics.jsonl").read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
