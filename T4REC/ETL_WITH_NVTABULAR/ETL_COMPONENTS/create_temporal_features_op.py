# Create temporal features with a `user-defined custom` op and `Lambda` op

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
    output_component_file="create_temporal_features_op.yaml"
)
def create_temporal_features_op(
    feat_list: list
):
    import os
    import numpy as np 
    import cupy as cp
    import glob
    import cudf
    import nvtabular as nvt
    from nvtabular import ColumnSelector

    # create time features
    session_ts = ['event_time_ts']

    session_time = (
        session_ts >> 
        nvt.ops.LambdaOp(lambda col: cudf.to_datetime(col, unit='s')) >> 
        nvt.ops.Rename(name = 'event_time_dt')
    )

    sessiontime_weekday = (
        session_time >> 
        nvt.ops.LambdaOp(lambda col: col.dt.weekday) >> 
        nvt.ops.Rename(name ='et_dayofweek')
    )

    def get_cycled_feature_value_sin(col, max_value):
        value_scaled = (col + 0.000001) / max_value
        value_sin = np.sin(2*np.pi*value_scaled)
        return value_sin

    def get_cycled_feature_value_cos(col, max_value):
        value_scaled = (col + 0.000001) / max_value
        value_cos = np.cos(2*np.pi*value_scaled)
        return value_cos

    weekday_sin = sessiontime_weekday >> (lambda col: get_cycled_feature_value_sin(col+1, 7)) >> nvt.ops.Rename(name = 'et_dayofweek_sin')
    weekday_cos= sessiontime_weekday >> (lambda col: get_cycled_feature_value_cos(col+1, 7)) >> nvt.ops.Rename(name = 'et_dayofweek_cos')

    from nvtabular.ops import Operator

    class ItemRecency(Operator):
        def transform(self, columns, gdf):
            for column in columns.names:
                col = gdf[column]
                item_first_timestamp = gdf['prod_first_event_time_ts']
                delta_days = (col - item_first_timestamp) / (60*60*24)
                gdf[column + "_age_days"] = delta_days * (delta_days >=0)
            return gdf
                
        def output_column_names(self, columns):
            return ColumnSelector([column + "_age_days" for column in columns.names])

        def dependencies(self):
            return ["prod_first_event_time_ts"]
        
        
    recency_features = ['event_time_ts'] >> ItemRecency() 
    recency_features_norm = recency_features >> nvt.ops.LogOp() >> nvt.ops.Normalize() >> nvt.ops.Rename(name='product_recency_days_log_norm')

    time_features = (
        session_time +
        sessiontime_weekday +
        weekday_sin +
        weekday_cos +
        recency_features_norm
    )