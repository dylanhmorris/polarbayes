#!/usr/bin/env python3

import polars as pl
import numpy as np

def spread_draws(posteriors, variable_names):
    """
    Given a dictionary of posteriors,
    return a long-form polars dataframe 
    indexed by draw, with variable
    values (equivalent of tidybayes
    spread_draws() function).
    """
    n_variables = len(variable_names)
        
    for i_var, v in enumerate(variable_names):
        if isinstance(v, str):
            v_dims = None
        else:
            v_dims = v[1:]
            v = v[0]
            
        post = posteriors.get(v)
        long = post.flatten()[..., np.newaxis]

        indices = np.array(list(np.ndindex(post.shape)))
        n_dims = indices.shape[1] - 1
        if v_dims is None:
            dim_names = [("{}_dim_{}_index".format(v, k),
                          pl.Int64)
                         for k in range(n_dims)]
        elif len(v_dims) != n_dims:
            raise ValueError("incorrect number of "
                             "dimension names "
                             "provided for variable "
                             "{}".format(v))
        else:
            dim_names = [(v_dim, pl.Int64)
                         for v_dim in v_dims]
        
        p_df = pl.DataFrame(
            np.concatenate(
                [indices, long],
                axis=1
            ),
        columns=[("draw", pl.Int64)] + dim_names + [(v, pl.Float64)])
        
        if i_var == 0:
            df = p_df
        else:
            df = df.join(p_df, 
                         on=[col for col in df.columns if col in p_df.columns])
        pass
    
    return df
