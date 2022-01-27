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
    output_component_file="export_parquet_from_bq_op.yaml"

)
def export_parquet_from_bq_op(
    output_dataset: Output[Dataset],
    bq_project: str,
    bq_location: str,
    bq_dataset_name: str,
    bq_table_name: str,
    split: str,
):
    '''
    Component to export PARQUET files from a bigquery table.

    output_datasets: dict
        Output metadata with the GCS path for the exported datasets.
    bq_project: str
        GCP project id
        Format:
            'my_project'
    bq_dataset_id: str
        Bigquery dataset id
        Format:
            'my_dataset_id'
    bq_table_train: str
        Bigquery table name for training dataset
        Format:
            'my_train_table_id'
    bq_table_valid: str
        BigQuery table name for validation dataset
        Format:
            'my_valid_table_id'
    '''

    import logging
    import os
    from preprocessing import etl
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)

    client = bigquery.Client(project=bq_project)
    dataset_ref = bigquery.DatasetReference(bq_project, bq_dataset_name)

    full_output_path = os.path.join(
        output_dataset.path.replace('/gcs/', 'gs://'),
        split
    )
    
    logging.info(
        f'Extracting {bq_table_name} table to {full_output_path} path.'
    )
    etl.extract_table_from_bq(
        client=client,
        output_dir=full_output_path,
        dataset_ref=dataset_ref,
        table_id=bq_table_name,
        location=bq_location
    )

    # Write metadata
    output_dataset.metadata['split'] = split

    logging.info('Finished exporting to GCS.')
