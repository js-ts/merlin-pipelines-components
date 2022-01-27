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
    output_component_file="transform_dataset_op.yaml"

  )
def transform_dataset_op(
    workflow: Input[Artifact],
    parquet_dataset: Input[Dataset],
    transformed_dataset: Output[Dataset],
    n_workers: int,
    shuffle: str = None,
    device_limit_frac: float = 0.8,
    device_pool_frac: float = 0.9,
    part_mem_frac: float = 0.125,
):
    '''
    Component to transform a dataset according to the workflow definitions.

    workflow: Input[Artifact]
        Input metadata with the path to the fitted_workflow
    parquet_dataset: Input[Dataset]
        Location of the converted dataset in GCS and split name
    transformed_dataset: Output[Dataset]
        Split name of the transformed dataset.
    n_workers: int
        Number of GPUs allocated to do the transformation
    '''
    from preprocessing import etl
    import logging
    import os

    logging.basicConfig(level=logging.INFO)

    # Create Dask cluster
    logging.info('Creating Dask cluster.')
    client = etl.create_cluster(
        n_workers=n_workers,
        device_limit_frac=device_limit_frac, 
        device_pool_frac=device_pool_frac
    )

    logging.info('Loading workflow and statistics')
    criteo_workflow = etl.load_workflow(
        workflow_path=workflow.path,
        client=client
    )

    split = parquet_dataset.metadata['split']

    logging.info(f'Creating dataset definition for {split} split')
    dataset = etl.create_parquet_dataset(
        client=client,
        data_path=os.path.join(
            parquet_dataset.path.replace('/gcs/', 'gs://'),
            split
        ),
        part_mem_frac=part_mem_frac
    )

    logging.info('Workflow is loaded')
    logging.info('Starting workflow transformation')
    dataset = etl.transform_dataset(
        dataset=dataset,
        workflow=criteo_workflow
    )

    logging.info('Applying transformation')
    etl.save_dataset(
        dataset, os.path.join(transformed_dataset.path, split)
    )

    transformed_dataset.metadata['split'] = split
