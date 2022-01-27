from collections import namedtuple
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
from typing import NamedTuple
# from . import config

@dsl.component(
    base_image="nvcr.io/nvidia/merlin/merlin-training:21.11",
    install_kfp_package=False,
    output_component_file="etl_with_nvtabular_op.yaml"
)


def etl_with_nvtabular_op(
# - Categorify categorical features with `Categorify()` op
    cat_feat_list: list = ['user_session', 'category_code', 'brand', 'user_id', 'product_id', 'category_id', 'event_type'],
    Data_Input: str="/dli/task/data/",
    session_feat_list: list=['event_time_ts']
)->NamedTuple('Output',[('output_path',str)]):
    # Import Libraries
    import os
    import numpy as np 
    import cupy as cp
    import glob
    import cudf
    import nvtabular as nvt
    from nvtabular import ColumnSelector
    from nvtabular.ops import Operator
    from numba import config        
    from preprocessing import etl
    import feature_utils
    from transformers4rec.data.preprocessing import save_time_based_splits
    
    # Set up Input and Output Data Paths
    INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR",Data_Input)
    # os.environ.get("INPUT_DATA_DIR", "/dli/task/data/")
    # Read the Input Parquet file
    df = cudf.read_parquet(os.path.join(INPUT_DATA_DIR, 'Oct-2019.parquet'))  
    
    
    # categorify features ['user_session', 'category_code', 'brand', 'user_id', 'product_id', 'category_id', 'event_type']
    # Initialize NVTabular Workflow

    # Categorical Features Encoding
    cat_feats = cat_feat_list >> nvt.ops.Categorify(start_index=1)



    # Create temporal features with a `user-defined custom` op and `Lambda` op


        # create time features
        # Extract Temporal Features
    session_ts = session_feat_list
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

    # Add Product Recency feature

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


    # - Transform continuous features using `Log` and `Normalize` ops


        # Smoothing price long-tailed distribution and applying standardization
    price_log = ['price'] >> nvt.ops.LogOp() >> nvt.ops.Normalize() >> nvt.ops.Rename(name='price_log_norm')
    # Normalize Continuous Features¶
    # Relative price to the average price for the category_id
    def relative_price_to_avg_categ(col, gdf):
        epsilon = 1e-5
        col = ((gdf['price'] - col) / (col + epsilon)) * (col > 0).astype(int)
        return col
        
    avg_category_id_pr = ['category_id'] >> nvt.ops.JoinGroupby(cont_cols =['price'], stats=["mean"]) >> nvt.ops.Rename(name='avg_category_id_price')
    relative_price_to_avg_category = avg_category_id_pr >> nvt.ops.LambdaOp(relative_price_to_avg_categ, dependency=['price']) >> nvt.ops.Rename(name="relative_price_to_avg_categ_id")


    # - Group all these features together at the session level sorting the interactions by time with `Groupby`


    
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
    # Grouping interactions into sessions
    # Aggregate by session id and creates the sequential features
    groupby_features_trim = groupby_features_list >> nvt.ops.ListSlice(0,SESSIONS_MAX_LENGTH) >> nvt.ops.Rename(postfix = '_seq')

    # calculate session day index based on 'timestamp-first' column
    day_index = ((groupby_features['event_time_dt-first'])  >> 
        nvt.ops.LambdaOp(lambda col: (col - col.min()).dt.days +1) >> 
        nvt.ops.Rename(f = lambda col: "day_index")
    )

    selected_features = groupby_features['user_session', 'product_id-count'] + groupby_features_trim + day_index

    filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["product_id-count"] >= MINIMUM_SESSION_LENGTH)

    # avoid numba warnings

    config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

    dataset = nvt.Dataset(df)
    workflow = nvt.Workflow(filtered_sessions)
    workflow.fit(dataset)
    sessions_gdf = workflow.transform(dataset).to_ddf()

    workflow_path = os.path.join(INPUT_DATA_DIR, 'workflow_etl')
    workflow.save(workflow_path)





        # define partition column
    PARTITION_COL = 'day_index'
    # make changes here use gcs upload component
    # define output_folder to store the partitioned parquet files
    OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", INPUT_DATA_DIR + "sessions_by_day")
    import subprocess
    subprocess.run(["mkdir", "-p",OUTPUT_FOLDER])

    save_time_based_splits(data=nvt.Dataset(sessions_gdf),
                       output_dir= OUTPUT_FOLDER,
                       partition_col=PARTITION_COL,
                       timestamp_col='user_session', 
                      )
    #implement named tuple
    from collections import namedtuple
    upload_file_path = namedtuple('Output',['OutputPath_Name'])
    return upload_file_path(OUTPUT_FOLDER)