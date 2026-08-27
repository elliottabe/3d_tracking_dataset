"""`_h5_safe`: segment stitching must produce a file stac_mjx can read back.

Segment stitching is the ONLY place run_bout loads an h5 written by stac_mjx
and writes it out again. stac_mjx stores strings as fixed-width BYTES
(`kp_names` as `|S12`, the YAML `config` as a `|S24514` scalar) and
`stac_mjx.io.load_stac_data` calls `.decode("utf-8")` on them. `ioh5.load`
returns those as numpy UNICODE arrays and Python `str`, so a naive round-trip
fails twice over:

  1. writing a `<U12` array at all -> TypeError: No conversion path for dtype
  2. writing it as a list/vlen-str instead -> the elements come back as `str`
     and blow up on `.decode` inside polish_bout

Both aborted the bout AFTER its ~6-minute pose solve had completed, so the
cost is a wasted solve rather than a wrong number. These assert the encoding
directly, against the access pattern load_stac_data really uses.
"""
from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


@pytest.fixture(scope="module")
def h5_safe():
    from scripts.run_bout import _h5_safe
    return _h5_safe


def test_numeric_arrays_pass_through_untouched(h5_safe):
    a = np.arange(6, dtype=np.float32).reshape(3, 2)
    assert h5_safe(a) is a


def test_non_string_objects_pass_through_untouched(h5_safe):
    """dicts/lists must NOT be stringified -- `config` travels through here."""
    d = {"model": {"a": 1}}
    assert h5_safe(d) is d
    lst = [1, 2, 3]
    assert h5_safe(lst) is lst


def test_unicode_array_becomes_a_fixed_width_bytes_array(h5_safe):
    out = h5_safe(np.array(["Scutellum", "WingL_base"]))
    assert isinstance(out, np.ndarray)
    assert out.dtype.kind == "S", f"expected bytes dtype, got {out.dtype}"
    assert [n.decode("utf-8") for n in out] == ["Scutellum", "WingL_base"]


def test_str_config_becomes_bytes(h5_safe):
    """load_stac_data does f['config'][()].decode('utf-8')."""
    out = h5_safe("model:\n  a: 1\n")
    assert isinstance(out, bytes)
    assert out.decode("utf-8") == "model:\n  a: 1\n"


def test_bytes_are_left_alone(h5_safe):
    b = np.array([b"Scutellum"])
    assert h5_safe(b) is b


def test_written_file_supports_load_stac_datas_access_pattern(h5_safe, tmp_path):
    """End-to-end on the two shapes that broke: prove the raw values are
    unwritable/unreadable and the converted ones satisfy `.decode`."""
    names = np.array(["Scutellum", "Abd_tip"])          # '<U9'
    cfg = "model:\n  a: 1\n"
    p = tmp_path / "stac_ik.h5"

    with h5py.File(p, "w") as f:                        # the raw value is rejected
        with pytest.raises(TypeError):
            f.create_dataset("kp_names", data=names)

    with h5py.File(p, "w") as f:
        f.create_dataset("kp_names", data=h5_safe(names))
        f.create_dataset("config", data=h5_safe(cfg))

    with h5py.File(p, "r") as f:
        assert [n.decode("utf-8") for n in f["kp_names"]] == ["Scutellum", "Abd_tip"]
        assert f["config"][()].decode("utf-8") == cfg
