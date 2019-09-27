import numpy as np
import torch
from torch import nn
import datetime
import time
import os
import tempfile
import unittest
from unittest import TestCase
from utils import (
    ensure_dir,
    generate_run_base_dir,
    count_params,
    load_args,
    save_args,
    set_cuda_device,
    clone_args,
    set_seed,
)
import argparse


class Test(TestCase):
    def test_ensure_dir(self):
        d = os.path.join(TMP, "a", "b", "c") + "/"
        ensure_dir(d)
        self.assertTrue(os.path.isdir(d))

    def test_generate_run_base_dir(self):
        res_dir = os.path.join(TMP, "res")
        t0 = time.time()
        tag = "tag"
        sub_dirs = ["a", "b", "c"]
        generate_run_base_dir(
            result_dir=res_dir, timestamp=t0, tag=tag, sub_dirs=sub_dirs
        )
        date_str = datetime.datetime.fromtimestamp(t0).strftime("%y%m%d_%H%M")
        self.assertTrue(
            os.path.isdir(os.path.join(res_dir, date_str + "_" + tag, *sub_dirs))
        )

    def test_count_params(self):
        linear = nn.Linear(123, 42)
        n_weights = 123 * 42
        n_bias = 42
        n_total = n_weights + n_bias
        self.assertEqual(n_total, count_params(linear))

    def test_load_save_args(self):
        parser = argparse.ArgumentParser()
        args = parser.parse_args(args=[])
        args.__dict__ = {"name": "test", "foo": "bar"}
        path = os.path.join(TMP, "args") + "/"
        ensure_dir(path)
        save_args(args, path)
        args_loaded = load_args(path)
        self.assertEqual(args, args_loaded)

    def test_clone_args(self):
        parser = argparse.ArgumentParser()
        args = parser.parse_args(args=[])
        args.__dict__ = {"name": "test", "foo": "bar"}

        cloned = clone_args(args)
        self.assertEqual(args.__dict__, cloned.__dict__)

    def test_set_cuda_device(self):
        set_cuda_device([0, 1, 2])
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1,2")

    def test_set_seed(self):
        seed = 42
        set_seed(seed)
        np_samples_a = np.random.randn(10)
        torch_samples_a = torch.randn(10)

        set_seed(seed)
        np_samples_b = np.random.randn(10)
        torch_samples_b = torch.randn(10)

        self.assertTrue(np.all(np_samples_a == np_samples_b))
        self.assertTrue(torch.all(torch_samples_a == torch_samples_b))


if __name__ == "__main__":
    TMP = tempfile.gettempdir()
    unittest.main()
