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
    output_component_file="gcs_downloading_objects.yaml",
    packages_to_install=['google-cloud-storage']
)
def gcs_downloading_objects(bucket_name: str, source_blob_name: str, destination_file_name: str
) -> NamedTuple('downloaded_bucket_values',[('dataset_name',str),('dataset_source_name',str),('download_file_name',str)]):
    from google.cloud import storage



    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )

    from collections import namedtuple
    donwloaded_file = namedtuple('downloaded_bucket_values',['dataset_name','dataset_source_name','download_file_name'])
    return donwloaded_file(bucket_name, source_blob_name, destination_file_name)