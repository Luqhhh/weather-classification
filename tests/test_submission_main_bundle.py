import importlib.util
import os
from pathlib import Path

import numpy as np
import torch


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = None

    def forward(self, x):
        return torch.zeros((x.shape[0], 768, 1, 1), dtype=torch.float32)


class FakeLinear(torch.nn.Module):
    calls = 0

    def __init__(self, in_features, out_features):
        super().__init__()
        self.fake_logits = torch.zeros(out_features, dtype=torch.float32)

    def forward(self, x):
        FakeLinear.calls += 1
        return self.fake_logits.to(x.device).repeat(x.shape[0], 1)


def _load_submission_main(tmp_path, monkeypatch, bundle):
    import torchvision.models as tv_models

    monkeypatch.setattr(tv_models, "convnext_tiny", lambda pretrained=False: FakeBackbone())
    monkeypatch.setattr(torch.nn, "Linear", FakeLinear)

    original_load_state_dict = torch.nn.Module.load_state_dict

    def fake_load_state_dict(self, state_dict, *args, **kwargs):
        if hasattr(self, "fc") and isinstance(self.fc, FakeLinear):
            self.fc.fake_logits = state_dict["fake_logits"].float().view(1, -1)
            return torch.nn.modules.module._IncompatibleKeys([], [])
        return original_load_state_dict(self, state_dict, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "load_state_dict", fake_load_state_dict)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    torch.save(bundle, results_dir / "convnext_tiny_320_best.pth")

    old_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        module_path = old_cwd / "天气识别" / "main.py"
        spec = importlib.util.spec_from_file_location(
            f"submission_main_{tmp_path.name}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(old_cwd)


def test_submission_main_applies_member_temperature_and_class_bias(tmp_path, monkeypatch):
    bundle = {
        "type": "logits_ensemble",
        "members": [
            {
                "state_dict": {"fake_logits": torch.tensor([2.0, 0.0, 0.0, 0.0])},
                "weight": 1.0,
                "temperature": 2.0,
                "image_size": 224,
            }
        ],
        "class_bias": [0.0, 1.5, 0.0, 0.0],
    }

    module = _load_submission_main(tmp_path, monkeypatch, bundle)

    prediction = module.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert prediction == "rainy"


def test_submission_main_runs_hflip_tta_views(tmp_path, monkeypatch):
    FakeLinear.calls = 0
    bundle = {
        "type": "logits_ensemble",
        "members": [
            {
                "state_dict": {"fake_logits": torch.tensor([0.0, 1.0, 0.0, 0.0])},
                "weight": 1.0,
                "image_size": 224,
            }
        ],
        "tta": ["identity", "hflip"],
    }

    module = _load_submission_main(tmp_path, monkeypatch, bundle)
    module.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert FakeLinear.calls == 2
