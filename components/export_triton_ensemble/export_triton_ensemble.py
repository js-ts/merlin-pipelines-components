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
      output_component_file="export_triton_ensemble.yaml"
)  
def export_triton_ensemble(
    model: Input[Model],
    workflow: Input[Artifact],
    exported_model: Output[Model],
    model_name: str,
    num_slots: int,
    max_nnz: int, 
    embedding_vector_size: int, 
    max_batch_size: int,
    model_repository_path: str
):
  
    import logging
    from serving import export
    import feature_utils
    
    logging.info('Exporting Triton ensemble model...')
    export.export_ensemble(
        model_name=model_name,
        workflow_path=workflow.path,
        saved_model_path=model.path,
        output_path=exported_model.path,
        categorical_columns=feature_utils.categorical_columns(),
        continuous_columns=feature_utils.continuous_columns(),
        label_columns=feature_utils.label_columns(),
        num_slots=num_slots,
        max_nnz=num_slots,
        num_outputs=max_nnz,
        embedding_vector_size=embedding_vector_size,
        max_batch_size=max_batch_size,
        model_repository_path=model_repository_path
    )
    logging.info('Triton model exported.')