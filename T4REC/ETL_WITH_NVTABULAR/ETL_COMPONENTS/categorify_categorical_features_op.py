# - Categorify categorical features with `Categorify()` op

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
    output_component_file="categorify_categorical_features_op.yaml"
)
def categorify_categorical_features_op(
    feat_list: list
):
    # import os
    # import numpy as np 
    # import cupy as cp
    # import glob
    # import cudf
    import nvtabular as nvt
    # from nvtabular import ColumnSelector
    # categorify features ['user_session', 'category_code', 'brand', 'user_id', 'product_id', 'category_id', 'event_type']
    cat_feats = feat_list >> nvt.ops.Categorify(start_index=1)