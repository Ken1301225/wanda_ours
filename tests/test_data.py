import importlib
import os
import sys
import unittest
from unittest.mock import patch


class DataPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = cls._import_data_module()

    @staticmethod
    def _import_data_module():
        torch_stub = object()
        datasets_stub = type("DatasetsStub", (), {"load_dataset": object()})()
        with patch.dict(sys.modules, {"torch": torch_stub, "datasets": datasets_stub}):
            return importlib.import_module("lib.data")

    def test_default_hf_hub_cache_uses_home_cache_dir(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(self.data.os.path, "expanduser", return_value="/tmp/home"):
            self.assertEqual(self.data._default_hf_hub_cache(), "/tmp/home/.cache/huggingface/hub")

    def test_default_hf_hub_cache_respects_environment_override(self):
        with patch.dict(os.environ, {"HF_HUB_CACHE": "/tmp/custom-cache"}, clear=True):
            self.assertEqual(self.data._default_hf_hub_cache(), "/tmp/custom-cache")

    def test_default_hf_hub_cache_uses_cache_root_when_available(self):
        with patch.dict(os.environ, {"HF_CACHE_ROOT": "/tmp/hf-cache-root"}, clear=True):
            self.assertEqual(self.data._default_hf_hub_cache(), "/tmp/hf-cache-root/hub")
