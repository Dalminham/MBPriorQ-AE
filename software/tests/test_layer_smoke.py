import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software/tools"))

from run_table2_ppl import _valid_layer_smoke


def _payload(layers=5):
    return {
        "mode": "mbpriorq_streamed_layer_smoke",
        "status": "PASS",
        "requested_layers": layers,
        "completed_layers": layers,
        "quantized_weight_count": 35,
        "wrapped_linear_count": 35,
        "layers": [
            {"layer_index": index, "output_finite": True}
            for index in range(layers)
        ],
    }


def test_valid_layer_smoke_requires_all_requested_layers(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(json.dumps(_payload()), encoding="utf-8")

    assert _valid_layer_smoke(result, 5)
    assert not _valid_layer_smoke(result, 4)


def test_valid_layer_smoke_rejects_nonfinite_layer(tmp_path):
    payload = _payload()
    payload["layers"][3]["output_finite"] = False
    result = tmp_path / "result.json"
    result.write_text(json.dumps(payload), encoding="utf-8")

    assert not _valid_layer_smoke(result, 5)
