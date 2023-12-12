import pytest

from dynamic_green_ammonia.technologies.chemical import HaberBosch


@pytest.fixture
def HB():
    return HaberBosch(1, 10)


class TestHaberBosch:
    def test__init__(self, HB):
        assert HB.dt == 1

    def test_step(self, HB):
        NH3, reject = HB.step(1, 1, 1)
        assert NH3 > 0
        assert (reject > 0).any()

    def test_calc_financials(self):
        pass
