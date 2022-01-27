# - Transform continuous features using `Log` and `Normalize` ops
# - Group all these features together at the session level sorting the interactions by time with `Groupby`
# - Finally export the preprocessed datasets to parquet files by hive-partitioning.


from kfp.v2 import dsl
from kfp.v2.dsl import (
    Artifact, 
    Dataset, 
    Input, 
    InputPath, 
    Model, 
    Output,
    OutputPath
)

from typing import Optional
# from . import config

@dsl.component(
    base_image="nvcr.io/nvidia/merlin/merlin-training:21.09",
    install_kfp_package=False,
    output_component_file="transform_continuous_features_op.yaml"
)
def transform_continuous_features_op(
    feat_list: list
):
    import os
    import numpy as np 
    import cupy as cp
    import glob
    import cudf
    import nvtabular as nvt
    from nvtabular import ColumnSelector
    # Smoothing price long-tailed distribution and applying standardization
    price_log = ['price'] >> nvt.ops.LogOp() >> nvt.ops.Normalize() >> nvt.ops.Rename(name='price_log_norm')

    # Relative price to the average price for the category_id
    def relative_price_to_avg_categ(col, gdf):
        epsilon = 1e-5
        col = ((gdf['price'] - col) / (col + epsilon)) * (col > 0).astype(int)
        return col
        
    avg_category_id_pr = ['category_id'] >> nvt.ops.JoinGroupby(cont_cols =['price'], stats=["mean"]) >> nvt.ops.Rename(name='avg_category_id_price')
    relative_price_to_avg_category = avg_category_id_pr >> nvt.ops.LambdaOp(relative_price_to_avg_categ, dependency=['price']) >> nvt.ops.Rename(name="relative_price_to_avg_categ_id")

