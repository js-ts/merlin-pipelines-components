# - Group all these features together at the session level sorting the interactions by time with `Groupby`

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
    output_component_file="group_all_features_with_groupby.yaml"
)
def group_all_features_with_groupby(
    feat_list: list
):
    import os
    import numpy as np 
    import cupy as cp
    import glob
    import cudf
    import nvtabular as nvt
    from nvtabular import ColumnSelector
    groupby_feats = ['event_time_ts', 'user_session'] + cat_feats + time_features + price_log + relative_price_to_avg_category

        # Define Groupby Workflow
    groupby_features = groupby_feats >> nvt.ops.Groupby(
        groupby_cols=["user_session"], 
        sort_cols=["event_time_ts"],
        aggs={
            'user_id': ['first'],
            'product_id': ["list", "count"],
            'category_code': ["list"],  
            'event_type': ["list"], 
            'brand': ["list"], 
            'category_id': ["list"], 
            'event_time_ts': ["first"],
            'event_time_dt': ["first"],
            'et_dayofweek_sin': ["list"],
            'et_dayofweek_cos': ["list"],
            'price_log_norm': ["list"],
            'relative_price_to_avg_categ_id': ["list"],
            'product_recency_days_log_norm': ["list"]
            },
        name_sep="-")

    groupby_features_list = groupby_features['product_id-list',
        'category_code-list',  
        'event_type-list', 
        'brand-list', 
        'category_id-list', 
        'et_dayofweek_sin-list',
        'et_dayofweek_cos-list',
        'price_log_norm-list',
        'relative_price_to_avg_categ_id-list',
        'product_recency_days_log_norm-list']

    
    SESSIONS_MAX_LENGTH = 20 
    MINIMUM_SESSION_LENGTH = 2

    groupby_features_trim = groupby_features_list >> nvt.ops.ListSlice(0,SESSIONS_MAX_LENGTH) >> nvt.ops.Rename(postfix = '_seq')

    # calculate session day index based on 'timestamp-first' column
    day_index = ((groupby_features['event_time_dt-first'])  >> 
        nvt.ops.LambdaOp(lambda col: (col - col.min()).dt.days +1) >> 
        nvt.ops.Rename(f = lambda col: "day_index")
    )

    selected_features = groupby_features['user_session', 'product_id-count'] + groupby_features_trim + day_index

    filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["product_id-count"] >= MINIMUM_SESSION_LENGTH)

    # avoid numba warnings
    from numba import config
    config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

    dataset = nvt.Dataset(df)
    workflow = nvt.Workflow(filtered_sessions)
    workflow.fit(dataset)
    sessions_gdf = workflow.transform(dataset).to_ddf()

    workflow_path = os.path.join(INPUT_DATA_DIR, 'workflow_etl')
    workflow.save(workflow_path)