import pytest
from app.services.preset_service import preset_service
from app.services.project_service import project_service

def test_iris_preset_download_and_caching():
    # 1. Download/build Iris preset
    res1 = preset_service.ensure_preset_downloaded("iris")
    assert res1["dataset_path"] is not None
    assert res1["model_path"] is not None

    # 2. Check second call uses cache
    res2 = preset_service.ensure_preset_downloaded("iris")
    assert res2["cached"] is True

def test_load_preset_into_project():
    proj = project_service.create_project(name="Iris Preset Test")
    load_res = preset_service.load_preset_into_project(proj.id, "iris")
    assert load_res["status"] == "success"
    assert load_res["preset_id"] == "iris"
    assert load_res["model"]["framework"] == "sklearn"
    assert load_res["dataset"]["version"] == "v001"

