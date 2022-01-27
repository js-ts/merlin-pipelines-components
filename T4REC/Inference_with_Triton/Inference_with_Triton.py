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
    base_image="nvcr.io/nvidia/merlin/merlin-inference:21.11",
    install_kfp_package=False,
    output_component_file="inference_with_triton.yaml"
)
def inference_with_triton(
    Model_Name: str = "t4r_pytorch",
    Data_Input: str="/dli/task/data/",
    ):
    # Import dependencies
    import os
    from time import time

    import argparse
    import numpy as np
    import pandas as pd
    import sys
    import cudf
    import tritonhttpclient

    try:
        triton_client = tritonhttpclient.InferenceServerClient(url="triton:8000", verbose=True)
        print("client created.")
    except Exception as e:
        print("channel creation failed: " + str(e))
    triton_client.is_server_live()

    triton_client.get_model_repository_index()

    triton_client.load_model(model_name=Model_Name)
   
   #comment down
   
    INPUT_DATA_DIR = Data_Input
    df= cudf.read_parquet(os.path.join(INPUT_DATA_DIR, 'Oct-2019.parquet'))
    df=df.sort_values('event_time_ts')
    batch = df.iloc[:50,:]

    sessions_to_use = batch.user_session.value_counts()
    filtered_batch = batch[batch.user_session.isin(sessions_to_use[sessions_to_use.values>1].index.values)]

    import warnings

    warnings.filterwarnings("ignore")

    import nvtabular.inference.triton as nvt_triton
    import tritonclient.grpc as grpcclient

    inputs = nvt_triton.convert_df_to_triton_input(filtered_batch.columns, filtered_batch, grpcclient.InferInput)

    output_names = ["output"]

    outputs = []
    for col in output_names:
        outputs.append(grpcclient.InferRequestedOutput(col))
        
    MODEL_NAME_NVT = "t4r_pytorch"

    with grpcclient.InferenceServerClient("triton:8001") as client:
        response = client.infer(MODEL_NAME_NVT, inputs)
        print(col, ':\n', response.as_numpy(col))