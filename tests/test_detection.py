"""
Tests for ragweed_toolkit.detection module.

Covers TrainingConfig defaults (Table 3) and GPS extraction from mock EXIF data.
Does NOT test actual YOLO training (requires GPU + ultralytics).
"""

import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import pytest

# TrainingConfig is a pure dataclass but the module does
# `from ultralytics import YOLO` at the top level.  Mock ultralytics
# so we can import just the dataclass even when ultralytics is absent.
HAS_TRAINER = False
try:
    from ragweed_toolkit.detection.trainer import TrainingConfig
    HAS_TRAINER = True
except ImportError:
    try:
        _fake = types.ModuleType("ultralytics")
        _fake.YOLO = type("YOLO", (), {})
        with mock.patch.dict(sys.modules, {"ultralytics": _fake}):
            from ragweed_toolkit.detection.trainer import TrainingConfig
        HAS_TRAINER = True
    except Exception:
        pass

try:
    from ragweed_toolkit.detection.gps import _dms_to_decimal, extract_gps
    HAS_GPS = True
except ImportError:
    HAS_GPS = False


# ------------------------------------------------------------------ #
# TrainingConfig defaults  (Chapter Table 3)
# ------------------------------------------------------------------ #

@pytest.mark.skipif(not HAS_TRAINER, reason="ultralytics not installed")
class TestTrainingConfigDefaults:
    """Verify that TrainingConfig defaults match the Springer chapter Table 3."""

    def test_max_det(self):
        cfg = TrainingConfig()
        assert cfg.max_det == 10_000

    def test_seed(self):
        cfg = TrainingConfig()
        assert cfg.seed == 42

    def test_box_loss(self):
        cfg = TrainingConfig()
        assert cfg.box == 7.5

    def test_cls_loss(self):
        cfg = TrainingConfig()
        assert cfg.cls == 0.5

    def test_dfl_loss(self):
        cfg = TrainingConfig()
        assert cfg.dfl == 1.5

    def test_lr0(self):
        cfg = TrainingConfig()
        assert cfg.lr0 == 0.01

    def test_momentum(self):
        cfg = TrainingConfig()
        assert cfg.momentum == 0.937

    def test_epochs(self):
        cfg = TrainingConfig()
        assert cfg.epochs == 50

    def test_imgsz(self):
        cfg = TrainingConfig()
        assert cfg.imgsz == 640

    def test_batch(self):
        cfg = TrainingConfig()
        assert cfg.batch == 64

    def test_optimizer(self):
        cfg = TrainingConfig()
        assert cfg.optimizer == "SGD"

    def test_amp(self):
        cfg = TrainingConfig()
        assert cfg.amp is True

    def test_override(self):
        cfg = TrainingConfig(epochs=100, lr0=0.001, device="cpu")
        assert cfg.epochs == 100
        assert cfg.lr0 == 0.001
        assert cfg.device == "cpu"


# ------------------------------------------------------------------ #
# GPS extraction
# ------------------------------------------------------------------ #

def _build_gps_exif_jpeg(lat_dms, lat_ref, lon_dms, lon_ref, altitude=None):
    """Create a minimal JPEG with GPS EXIF data using Pillow.

    Returns a temporary file path.
    """
    import piexif
    from PIL import Image

    # Create a small image
    img = Image.new("RGB", (10, 10), color="green")

    # Build GPS IFD
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: lat_ref.encode(),
        piexif.GPSIFD.GPSLatitude: _to_rational_tuple(lat_dms),
        piexif.GPSIFD.GPSLongitudeRef: lon_ref.encode(),
        piexif.GPSIFD.GPSLongitude: _to_rational_tuple(lon_dms),
    }
    if altitude is not None:
        gps_ifd[piexif.GPSIFD.GPSAltitude] = (int(altitude * 100), 100)
        gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = 0

    exif_dict = {"GPS": gps_ifd, "0th": {}, "Exif": {}, "1st": {}}
    exif_bytes = piexif.dump(exif_dict)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, exif=exif_bytes)
    return tmp.name


def _to_rational_tuple(dms):
    """Convert (deg, min, sec) to piexif rational tuple."""
    d, m, s = dms
    return ((int(d), 1), (int(m), 1), (int(s * 10000), 10000))


HAS_PIEXIF = True
try:
    import piexif  # noqa: F401
except ImportError:
    HAS_PIEXIF = False


@pytest.mark.skipif(not HAS_GPS, reason="PIL not available")
@pytest.mark.skipif(not HAS_PIEXIF, reason="piexif not installed")
class TestGPSExtractionWithExif:
    """Test GPS coordinate extraction from mock EXIF data (requires piexif)."""

    def test_southern_hemisphere(self):
        """INIA fields are in Chile: ~34S, ~71W."""
        path = _build_gps_exif_jpeg(
            lat_dms=(34, 10, 30.0), lat_ref="S",
            lon_dms=(71, 5, 15.0), lon_ref="W",
            altitude=312.0,
        )
        result = extract_gps(path)
        assert result is not None
        assert result["latitude"] == pytest.approx(-34.175, abs=0.01)
        assert result["longitude"] == pytest.approx(-71.0875, abs=0.01)
        Path(path).unlink()

    def test_northern_hemisphere(self):
        path = _build_gps_exif_jpeg(
            lat_dms=(47, 30, 0.0), lat_ref="N",
            lon_dms=(19, 3, 0.0), lon_ref="E",
        )
        result = extract_gps(path)
        assert result is not None
        assert result["latitude"] > 0
        assert result["longitude"] > 0
        Path(path).unlink()


@pytest.mark.skipif(not HAS_GPS, reason="PIL not available")
class TestGPSNoExif:
    """Test GPS extraction on images without GPS data (no piexif needed)."""

    def test_no_gps_returns_none(self):
        """An image without GPS EXIF should return None."""
        from PIL import Image

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img = Image.new("RGB", (10, 10), color="red")
        img.save(tmp.name)
        result = extract_gps(tmp.name)
        assert result is None
        Path(tmp.name).unlink()


@pytest.mark.skipif(not HAS_GPS, reason="PIL not available")
class TestDmsToDecimal:
    """Test the DMS -> decimal conversion helper."""

    def test_south(self):
        assert _dms_to_decimal((34, 10, 30.0), "S") == pytest.approx(-34.175, abs=0.001)

    def test_north(self):
        assert _dms_to_decimal((47, 30, 0.0), "N") == pytest.approx(47.5, abs=0.001)

    def test_west(self):
        assert _dms_to_decimal((71, 0, 0.0), "W") == pytest.approx(-71.0, abs=0.001)

    def test_east(self):
        assert _dms_to_decimal((19, 3, 0.0), "E") == pytest.approx(19.05, abs=0.001)
