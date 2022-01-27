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
    output_component_file="export_data.yaml"
)
def export_data(
    feat_list: list
):
    import os
    import numpy as np 
    import cupy as cp
    import glob
    import cudf
    import nvtabular as nvt
    from nvtabular import ColumnSelector
    import logging
    import os
    from preprocessing import etl
    import feature_utils
    from transformers4rec.data.preprocessing import save_time_based_splits
    save_time_based_splits(data=nvt.Dataset(sessions_gdf),
                        output_dir= OUTPUT_FOLDER,
                        partition_col=PARTITION_COL,
                        timestamp_col='user_session', 
                        )