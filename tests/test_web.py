"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))  # the repo folder is the thunderstorm_python package

from thunderstorm_python.web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific ---------------------------------------------------------------------------------------------
import io  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from thunderstorm_python.pipeline import ThunderSTORM  # noqa: E402
from thunderstorm_python.web.demo import CAMERA, SIM, simulate  # noqa: E402


def test_info_defaults_come_from_the_package():
    d = client.get("/api/info").json()["defaults"]
    assert d["filter_type"] == "wavelet" and d["filter_params"] == {"scale": 2, "order": 3}
    assert d["threshold_expression"] == "std(Wave.F1)" and d["fit_radius"] == 3 and d["pixel_size"] == 100.0
    opts = client.get("/api/info").json()["options"]
    assert {o["id"] for o in opts["fitter"]} == {"gaussian_lsq", "gaussian_mle", "radial_symmetry", "centroid"}


def test_demo_movie_is_reproducible_and_labelled():
    m = client.get("/api/demo/movie").json()
    assert m["simulated"] is True and "Not real data" in m["description"]
    assert m["shape"] == [SIM["n_frames"], SIM["image_size"], SIM["image_size"]]
    simulate.cache_clear()
    again = client.get("/api/demo/movie").json()
    assert again["frames_b64"] == m["frames_b64"] and again["truth"] == m["truth"]
    assert len(m["truth"]["x"]) == client.get("/api/info").json()["demo"]["n_emitters"]


def test_demo_result_matches_direct_pipeline_call():
    r = client.get("/api/demo/result").json()
    movie = simulate()[0]
    pipe = ThunderSTORM(pixel_size=CAMERA["pixel_size"], photons_per_adu=1.0, baseline=0.0)
    locs = pipe.analyze_stack(movie.astype(float), show_progress=False)
    assert r["n"] == len(locs["x"]) and r["pixel_size_nm"] == 100.0
    np.testing.assert_allclose(r["locs"]["x"], locs["x"], rtol=1e-12)
    ev = r["evaluation"]
    assert ev["n_gt"] == len(client.get("/api/demo/movie").json()["truth"]["x"])
    assert 0.5 < ev["recall"] <= 1 and 0.5 < ev["precision"] <= 1 and ev["rmse_nm"] < 60


def _tiff_bytes(arr, **kw):
    import tifffile
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, **kw)
    return buf.getvalue()


def test_upload_tiff_uses_file_pixel_size_and_frame_endpoint():
    movie = simulate()[0][:5]
    data = _tiff_bytes(movie, imagej=True, resolution=(1 / 0.16, 1 / 0.16), metadata={"unit": "micron", "axes": "TYX"})
    params = {**client.get("/api/info").json()["defaults"], "pixel_size": 100.0, "baseline": 0.0}
    r = client.post("/api/analyze", data={"params": json.dumps(params)}, files={"file": ("stack.tif", data, "image/tiff")})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["pixel_size_source"] == "file metadata" and abs(res["pixel_size_nm"] - 160.0) < 1e-6
    direct = ThunderSTORM(pixel_size=160.0, baseline=0.0).analyze_stack(movie.astype(float), show_progress=False)
    np.testing.assert_allclose(res["locs"]["x"], direct["x"], rtol=1e-12)
    fr = client.get(f"/api/runs/{res['run_id']}/frame/2").json()
    assert fr["shape"] == [64, 64]
    # the user's value wins when asked to ignore the file
    params["use_file_pixel_size"] = False
    res2 = client.post("/api/analyze", data={"params": json.dumps(params)}, files={"file": ("stack.tif", data, "image/tiff")}).json()
    assert res2["pixel_size_nm"] == 100.0 and res2["pixel_size_source"] == "user" and abs(res2["file_pixel_size_nm"] - 160) < 1e-6


def test_upload_errors():
    params = json.dumps(client.get("/api/info").json()["defaults"])
    bad = client.post("/api/analyze", data={"params": params}, files={"file": ("x.tif", b"not a tiff", "image/tiff")})
    assert bad.status_code == 400
    rgb = _tiff_bytes(np.zeros((2, 3, 20, 20), np.uint16))
    assert client.post("/api/analyze", data={"params": params}, files={"file": ("x.tif", rgb, "image/tiff")}).status_code == 400
    p = json.loads(params)
    p["threshold_expression"] = "std(Nope.F9)"
    assert client.post("/api/analyze", data={"params": json.dumps(p)}).status_code == 400
    p["threshold_expression"], p["filter_type"] = "std(Wave.F1)", "magic"
    assert client.post("/api/analyze", data={"params": json.dumps(p)}).status_code == 400
    assert client.get("/api/runs/nope/frame/0").status_code == 404


def test_engine_fixture_is_current():
    """tests/fixtures/engine_cases.json (read by ts_engine.test.mjs) still matches the Python pipeline."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_engine_fixture
    fresh = make_engine_fixture.build()
    stored = json.loads((ROOT / "tests" / "fixtures" / "engine_cases.json").read_text())

    def close(a, b, path="fixture"):
        if isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for k in a:
                close(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list):
            assert len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                close(x, y, f"{path}[{i}]")
        elif isinstance(a, float) and isinstance(b, (int, float)):
            assert abs(a - b) <= 1e-6 + 1e-6 * abs(b), f"{path}: {a} vs {b}"
        else:
            assert a == b, f"{path}: {a} vs {b}"
    close(fresh, stored)
