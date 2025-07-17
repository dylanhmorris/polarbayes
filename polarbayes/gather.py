from typing import Iterable

import arviz as az
import numpy as np
import polars as pl

from polarbayes.spread import spread_draws_and_get_index_cols


def _assert_not_in_index_columns(arg_name, arg_value, index_columns) -> None:
    if arg_value in index_columns:
        raise ValueError(
            f"Specified {arg_name}='{arg_value}' for the output data frame "
            f"but there is an index column named '{arg_value}' "
            f"in the input data frame. Either specify a different "
            f" {arg_name} or rename the index column named '{arg_value}'."
        )


def gather_variables(
    data: pl.DataFrame,
    index_cols=[],
    exclude=["chain", "draw"],
    value_name="value",
    variable_name="variable",
):
    """ """
    index = sorted(set(index_cols).union(set(exclude)))

    # more informative error message than `unpivot()` gives on its own
    [
        _assert_not_in_index_columns(k, v, index)
        for k, v in dict(
            value_name=value_name, variable_name=variable_name
        ).items()
    ]

    return data.unpivot(
        index=index, variable_name=variable_name, value_name=value_name
    ).select(
        pl.col("chain"),
        pl.col("draw"),
        pl.all().exclude(["chain", "draw"]),
        # always start the df with chain, then draw, then any other index cols
        # in alphabetical order
    )


def gather_draws(
    data: az.InferenceData,
    group: str = "posterior",
    combined: bool = True,
    var_names: Iterable[str] = None,
    filter_vars: str = None,
    num_samples: int = None,
    rng: bool | int | np.random.Generator = None,
) -> pl.DataFrame:
    # need to extract all variables jointly to ensure same
    # draws for each
    extracted = az.extract(
        data,
        group=group,
        combined=combined,
        var_names=var_names,
        filter_vars=filter_vars,
        num_samples=num_samples,
        keep_dataset=True,
        rng=rng,
    )
    var_names = extracted.data_vars.keys()
    return pl.concat(
        [
            gather_variables(
                *spread_draws_and_get_index_cols(
                    extracted,
                    group=group,
                    var_names=var,
                    combined=False,
                    filter_vars=None,
                    num_samples=None,
                    rng=False,
                    enforce_drop_chain_draw=combined,
                )
            )
            for var in var_names
        ],
        how="diagonal",
    )
