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
from typing import NamedTuple
from typing import Optional
# from . import config

@dsl.component(
    base_image="python:3.7",
    install_kfp_package=False,
    output_component_file="gcs_uploading_objects.yaml",
    packages_to_install=['google-cloud-storage']
)
def gcs_uploading_objects(bucket_name, source_file_name, destination_blob_name
) -> NamedTuple('uploaded_bucket_values',[('dataset_name',str),('dataset_file_name',str),('upload_file_name',str)]):
    from google.cloud import storage
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

    from collections import namedtuple
    donwloaded_file = namedtuple('uploaded_bucket_values',['dataset_name','dataset_file_name','upload_file_name'])
    return donwloaded_file(bucket_name, source_file_name, destination_blob_name)