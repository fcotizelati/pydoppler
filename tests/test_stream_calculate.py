import numpy as np

from pydoppler import pydoppler as pydoppler_module


def test_stream_calculate_near_unity_keeps_ratio():
    xl_near, _, _, _, _, _ = pydoppler_module.stream_calculate(0.9999, ni=30, nj=30)
    xl_ref, _, _, _, _, _ = pydoppler_module.stream_calculate(0.999, ni=30, nj=30)

    assert np.allclose(xl_near, xl_ref, rtol=1e-6, atol=1e-6)
