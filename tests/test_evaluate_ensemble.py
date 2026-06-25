from PIL import Image

from data.label_mapping import LabelMapper
from scripts.evaluate_ensemble import _build_member_datasets


def test_build_member_datasets_uses_each_config_image_size(tmp_path):
    class_dir = tmp_path / "cloudy"
    class_dir.mkdir()
    Image.new("RGB", (80, 80), color=(120, 130, 140)).save(class_dir / "sample.jpg")

    label_mapper = LabelMapper(["cloudy"])
    configs = [
        {"data": {"image_size": 32}},
        {"data": {"image_size": 48}},
    ]

    datasets = _build_member_datasets(tmp_path, label_mapper, configs)

    assert len(datasets) == 2
    image_32, label_32 = datasets[0][0]
    image_48, label_48 = datasets[1][0]
    assert image_32.shape[-2:] == (32, 32)
    assert image_48.shape[-2:] == (48, 48)
    assert label_32 == label_48 == 0
