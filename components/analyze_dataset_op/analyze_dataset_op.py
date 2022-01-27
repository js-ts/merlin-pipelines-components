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
    output_component_file="analyze_dataset_op.yaml"
)
def analyze_dataset_op(
    parquet_dataset: Input[Dataset],
    workflow: Output[Artifact],
    n_workers: int,
    device_limit_frac: Optional[float] = 0.8,
    device_pool_frac: Optional[float] = 0.9,
    part_mem_frac: Optional[float] = 0.125
):
    '''
    Component to generate statistics from the dataset.

    parquet_dataset: Input[Dataset]
        Input metadata with references to the train and valid converted
        datasets in GCS and the split name.
        Usage:
            parquet_dataset.path
            parquet_dataset.metadata['split']
    workflow: Output[Artifact]
        Output metadata with the path to the fitted workflow artifacts
        (statistics).
    n_workers: int
        Number of GPUs allocated to do the fitting process
    shuffle: str
        How to shuffle the transformed data, default to None.
        Options:
            PER_PARTITION
            PER_WORKER
            FULL
    '''
    from preprocessing import etl
    import logging
    import os

    logging.basicConfig(level=logging.INFO)

    split = parquet_dataset.metadata['split']

    # Create Dask cluster
    logging.info('Creating Dask cluster.')
    client = etl.create_cluster(
        n_workers = n_workers,
        device_limit_frac = device_limit_frac, 
        device_pool_frac = device_pool_frac
    )

    # Create data transformation workflow. This step will only 
    # calculate statistics based on the transformations
    logging.info('Creating transformation workflow.')
    criteo_workflow = etl.create_criteo_nvt_workflow(client=client)

    # Create dataset to be fitted
    logging.info(f'Creating dataset to be analysed.')
    logging.info(f'Base path in {parquet_dataset.path}')
    dataset = etl.create_parquet_dataset(
        client=client,
        data_path=os.path.join(
            parquet_dataset.path.replace('/gcs/','gs://'),
            split
        ),
        part_mem_frac=part_mem_frac
    )

    logging.info(f'Starting workflow fitting for {split} split.')
    criteo_workflow = etl.analyze_dataset(criteo_workflow, dataset)
    logging.info('Finished generating statistics for dataset.')

    etl.save_workflow(criteo_workflow, workflow.path)
    logging.info('Workflow saved to GCS')
