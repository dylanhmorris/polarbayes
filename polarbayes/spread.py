from typing import Iterable

import arviz as az
import numpy as np
import pandas as pd
import polars as pl


def spread_draws_to_pandas_(
    data: az.InferenceData,
    group: str = "posterior",
    combined: bool = True,
    var_names: Iterable[str] = None,
    filter_vars: str = None,
    num_samples: int = None,
    rng: bool | int | np.random.Generator = None,
) -> pd.DataFrame:
    """
    Convert an ArviZ InferenceData object group to a Pandas
    DataFrame of tidy (spread) draws, using the syntax of
    arviz.extract

    Parameters
    ----------
    data
        Data to convert.

    group
        `group` parameter passed to `az.extract`.

    combined
        `combined` parameter passed to `az.extract`.

    var_names
        `var_names` parameter passed to `az.extract`.

    filter_vars
        `var_names` parameter passed to `az.extract`.

    num_samples
        `num_samples` parameter passed to `az.extract`.

    rng
        `rng` parameter passed to `az.extract`.

    Returns
    -------
    pd.DataFrame
       Pandas DataFrame with a MultiIndex on the chain id,
       draw id, and any additional indices determined by
       the dimensions of the variables selected via
       `var_names` or `filter_vars`.
    """
    df = az.extract(
        data,
        group=group,
        combined=combined,
        var_names=var_names,
        filter_vars=filter_vars,
        num_samples=num_samples,
        keep_dataset=True,
        rng=rng,
    ).to_dataframe()
    if combined:
        # az.extract with combined=True assumes that the InferenceData object has
        # "chain" and "draw" named dimensions and errors otherwise.
        # When the resultant dataset is converted to a pandas dataframe via .to_dataframe(),
        # "chain" and "draw" are included _both_ as column names and as Pandas MultiIndex
        # names. We only want/need them in the MultiIndex, so we drop the columns, which
        # are guaranteed to exist per by the az.extract assumption mentioned above.
        df = df.drop(["chain", "draw"], axis=1)
    return df


def spread_draws_and_get_index_cols(
    data: az.InferenceData,
    group: str = "posterior",
    combined: bool = True,
    var_names: Iterable[str] = None,
    filter_vars: str = None,
    num_samples: int = None,
    rng: bool | int | np.random.Generator = None,
    enforce_drop_chain_draw: bool = False,
) -> tuple[pl.DataFrame, tuple]:
    """
    Convert an ArviZ InferenceData object to a polars
    DataFrame of tidy (spread) draws, using the syntax of
    arviz.extract. Return that DataFrame alongside a tuple
    giving the names of the DataFrame's index columns.

    Parameters
    ----------
    data
        Data to convert.

    group
        `group` parameter passed to `az.extract`.

    combined
        `combined` parameter passed to `az.extract`.

    var_names
        `var_names` parameter passed to `az.extract`.

    filter_vars
        `var_names` parameter passed to `az.extract`.

    num_samples
        `num_samples` parameter passed to `az.extract`.

    rng
        `rng` parameter passed to `az.extract`.

    Returns
    -------
    tuple[pl.DataFrame, tuple]
        Two-entry whose first entry is the DataFrame and whose
        second entry is a tuple giving the names of the DataFrame's
        index columns, typically `"chain"`, "draw"`, and additional
        columns determined by the dimensions of the variables chosen
        via `var_names` or `filter_vars`.
    """

    df = spread_draws_to_pandas_(
        data,
        group=group,
        combined=combined,
        var_names=var_names,
        filter_vars=filter_vars,
        num_samples=num_samples,
        rng=rng,
    )
    if enforce_drop_chain_draw:
        df = df.drop(["chain", "draw"], axis=1)
    return (pl.DataFrame(df.reset_index()), tuple(df.index.names))


def spread_draws(
    data: az.InferenceData,
    group: str = "posterior",
    combined: bool = True,
    var_names: Iterable[str] = None,
    filter_vars: str = None,
    num_samples: int = None,
    rng: bool | int | np.random.Generator = None,
) -> pl.DataFrame:
    result, _ = spread_draws_and_get_index_cols(
        data,
        group=group,
        combined=combined,
        var_names=var_names,
        filter_vars=filter_vars,
        num_samples=num_samples,
        rng=rng,
    )
    return result
