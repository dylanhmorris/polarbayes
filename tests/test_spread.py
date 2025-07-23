import arviz as az
import pandas as pd
import pytest

from polarbayes.spread import spread_draws_to_pandas_


@pytest.fixture(params=["rugby_field", "centered_eight"])
def idata(request):
    """
    InferenceData objects to test on
    """
    return az.load_arviz_data(request.param)


@pytest.mark.parametrize("combined", [True, False])
def test_spread_to_pandas_default(idata, combined):
    """
    Test that all variables present in the idata are present in extracted data
    unless a filter is requested, regardless of the value of `combined`.
    """
    index_vars = idata.posterior.coords.keys()
    assert "chain" in index_vars
    assert "draw" in index_vars
    expected_vars = [
        k for k in idata.posterior.variables.keys() if k not in index_vars
    ]
    result = spread_draws_to_pandas_(idata)
    for i_var in index_vars:
        assert i_var in result.index.names, f"Index name {i_var} not found"
        assert i_var not in result.columns, (
            f"Index name {i_var} should not be a column"
        )
    for var in expected_vars:
        assert var in result.columns, f"Variable {var} not found"
        assert var not in result.index.names, (
            f"Variable {var} should not form part of the index"
        )


def test_downsampling_reproducibility(idata):
    """
    Test that the `rng` parameter provides reproducible downsampling
    """
    sample_size = 50

    result1 = spread_draws_to_pandas_(idata, num_samples=sample_size, rng=42)
    result2 = spread_draws_to_pandas_(idata, num_samples=sample_size, rng=42)
    result3 = spread_draws_to_pandas_(idata, num_samples=sample_size, rng=43)

    pd.testing.assert_frame_equal(result1, result2)
    assert result1.shape == result3.shape
    assert not result1.equals(result3)
