"""Risk-score thresholds: OTP (0–1 scale) maps to Low/Medium/High."""
import pytest

from shared import data


class TestRiskLevel:
    @pytest.mark.parametrize(
        "otp,expected",
        [
            (1.00, "Low"),
            (0.97, "Low"),
            (0.95, "Low"),      # boundary: >= 0.95 is Low
            (0.9499, "Medium"),
            (0.92, "Medium"),
            (0.90, "Medium"),   # boundary: >= 0.90 is Medium
            (0.8999, "High"),
            (0.85, "High"),
            (0.0, "High"),
        ],
    )
    def test_thresholds(self, otp, expected):
        assert data.risk_level(otp) == expected

    def test_every_level_has_a_display_color(self):
        for level in ("Low", "Medium", "High"):
            assert level in data.RISK_COLOR
