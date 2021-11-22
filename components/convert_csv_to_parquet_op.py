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
    output_component_file="convert_csv_to_parquet_op.yaml"
)
def convert_csv_to_parquet_op(
    output_dataset: Output[Dataset],
    data_paths: list,
    split: str,
    sep: str,
    num_output_files: int,
    n_workers: int,
    shuffle: Optional[str] = None,
    recursive: Optional[bool] = False,
    device_limit_frac: Optional[float] = 0.8,
    device_pool_frac: Optional[float] = 0.9,
    part_mem_frac: Optional[float] = 0.125
):
    '''
    Component to convert CSV file(s) to Parquet format using NVTabular.

    output_dataset: Output[Dataset]
        Output metadata with references to the converted CSV files in GCS
        and the split name.
        The path to the files are in GCS fuse format:
            /gcs/<bucket name>/path/to/file
        Usage:
            output_dataset.path
            output_dataset.metadata['split']
    data_paths: list
        List of paths to folders or files on GCS.
        For recursive folder search, set the recursive variable to True
        Format:
            'gs://<bucket_name>/<subfolder1>/<subfolder>/' or
            'gs://<bucket_name>/<subfolder1>/<subfolder>/flat_file.csv' or
            a combination of both.
    split: str
        Split name of the dataset. Example: train or valid
    n_workers: int
        Number of GPUs allocated to convert the CSV to Parquet
    shuffle: str
        How to shuffle the converted CSV, default to None.
        Options:
            PER_PARTITION
            PER_WORKER
            FULL
    recursive: bool
        Recursivelly search for files in path.
    '''
    
    import logging
    import os
    from preprocessing import etl
    import feature_utils

    logging.basicConfig(level=logging.INFO)

    logging.info('Getting column names and dtypes')
    col_dtypes = feature_utils.get_criteo_col_dtypes()

    # Create Dask cluster
    logging.info('Creating Dask cluster.')
    client = etl.create_cluster(
        n_workers = n_workers,
        device_limit_frac = device_limit_frac,
        device_pool_frac = device_pool_frac
    )

    logging.info(f'Creating {split} dataset.')
    dataset = etl.create_csv_dataset(
        data_paths=data_paths,
        sep=sep,
        recursive=recursive, 
        col_dtypes=col_dtypes,
        part_mem_frac=part_mem_frac, 
        client=client
    )

    logging.info(f'Base path in {output_dataset.path}')
    fuse_output_dir = os.path.join(output_dataset.path, split)
    
    logging.info(f'Writing parquet file(s) to {fuse_output_dir}')
    etl.convert_csv_to_parquet(
        output_path=fuse_output_dir,
        dataset=dataset, 
        output_files=num_output_files, 
        shuffle=shuffle
    )

    # Write metadata
    output_dataset.metadata['split'] = split


