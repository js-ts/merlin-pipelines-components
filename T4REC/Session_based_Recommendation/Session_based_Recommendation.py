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
    base_image="nvcr.io/nvidia/merlin/merlin-pytorch-training:21.11",
    install_kfp_package=False,
    output_component_file="session_based_recsys.yaml"
)
def session_based_recsys(    
    Output_Model_Path: str="/dli/task/model_repository",
    Output_Model_Name: str = "t4r_pytorch",
    Data_Input: str="/dli/task/data/",
    Data_Output: str = "/dli/task/data/sessions_by_day",
    local_rank : int= -1,
    training_args_output_dir: str="./tmp",
    training_args_max_sequence_length: int=20,
    training_args_data_loader_engine: str='nvtabular',
    training_args_num_train_epochs: int=3, 
    training_args_dataloader_drop_last=False,
    training_args_per_device_train_batch_size : int= 256,
    training_args_per_device_eval_batch_size : int= 32,
    training_args_gradient_accumulation_steps : int= 1,
    training_args_learning_rate: int=0.000666,
    training_args_report_to :list= [],
    training_args_logging_steps: int=200,
)->NamedTuple('Model',[('model',str)]):
    # Training an RNN-based Session-based Recommendation Model
    import os
    import glob

    import torch 
    import transformers4rec.torch as tr

    from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt
    from transformers4rec.torch.utils.examples_utils import wipe_memory

    # Instantiates Schema object from a `schema` file.

    from merlin_standard_lib import Schema
    # Define schema object to pass it to the TabularSequenceFeatures class
    INPUT_DATA_DIR = Data_Input

    SCHEMA_PATH = os.path.join(INPUT_DATA_DIR, 'schema_tutorial.pb')
    schema = Schema().from_proto_text(SCHEMA_PATH)
    schema = schema.select_by_name(['product_id-list_seq'])

    # Defining the input block: `TabularSequenceFeatures`

    # Define input block
    sequence_length = 20
    inputs = tr.TabularSequenceFeatures.from_schema(
            schema,
            max_sequence_length= sequence_length,
            masking = 'causal',
        )

    # Connecting the blocks with `SequentialBlock`
    
    d_model = 128
    body = tr.SequentialBlock(
            inputs,
            tr.MLPBlock([d_model]),
            tr.Block(torch.nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1), [None, 20, d_model])
    )

    # Item Prediction head and tying embeddings

    head = tr.Head(
    body,
        tr.NextItemPredictionTask(weight_tying=True, hf_format=True, 
                                metrics=[NDCGAt(top_ks=[10, 20], labels_onehot=True),  
                                        RecallAt(top_ks=[10, 20], labels_onehot=True)]),
    )
    model = tr.Model(head)

    # Define a Dataloader function from schema

    # import NVTabular dependencies
    from transformers4rec.torch.utils.data_utils import NVTabularDataLoader

    x_cat_names, x_cont_names = ['product_id-list_seq'], []

    # dictionary representing max sequence length for column
    sparse_features_max = {
        fname: sequence_length
        for fname in x_cat_names + x_cont_names
    }

    # Define a `get_dataloader` function to call in the training loop
    def get_dataloader(path, batch_size=32):

        return NVTabularDataLoader.from_schema(
            schema,
            path, 
            batch_size,
            max_sequence_length=sequence_length,
            sparse_names=x_cat_names + x_cont_names,
            sparse_max=sparse_features_max,
    )

    # Daily Fine-Tuning: Training over a time window

    from transformers4rec.config.trainer import T4RecTrainingArguments
    from transformers4rec.torch import Trainer

    #Set arguments for training 
    train_args = T4RecTrainingArguments(local_rank = local_rank, 
                                        dataloader_drop_last = training_args_dataloader_drop_last,
                                        report_to = training_args_report_to,   #set empy list to avoig logging metrics to Weights&Biases
                                        gradient_accumulation_steps = training_args_gradient_accumulation_steps,
                                        per_device_train_batch_size = training_args_per_device_train_batch_size, 
                                        per_device_eval_batch_size = training_args_per_device_eval_batch_size,
                                        output_dir = training_args_output_dir, 
                                        max_sequence_length=sequence_length,
                                        learning_rate=0.00071,
                                        num_train_epochs=training_args_num_train_epochs,
                                        logging_steps=training_args_logging_steps,
                                    )

    # Instantiate the T4Rec Trainer, which manages training and evaluation
    trainer = Trainer(
        model=model,
        args=train_args,
        schema=schema,
        compute_metrics=True,
    )
    #recieve input from the 
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", Data_Output)

    # os.system("%%time")

    start_time_window_index = 1
    final_time_window_index = 4
    for time_index in range(start_time_window_index, final_time_window_index):
        # Set data 
        time_index_train = time_index
        time_index_eval = time_index + 1
        train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))
        eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))
        # Train on day related to time_index 
        print('*'*20)
        print("Launch training for day %s are:" %time_index)
        print('*'*20 + '\n')
        trainer.train_dataset_or_path = train_paths
        trainer.reset_lr_scheduler()
        trainer.train()
        trainer.state.global_step +=1
        # Evaluate on the following day
        trainer.eval_dataset_or_path = eval_paths
        train_metrics = trainer.evaluate(metric_key_prefix='eval')
        print('*'*20)
        print("Eval results for day %s are:\t" %time_index_eval)
        print('\n' + '*'*20 + '\n')
        for key in sorted(train_metrics.keys()):
            print(" %s = %s" % (key, str(train_metrics[key]))) 
        wipe_memory()

        from collections import namedtuple
        upload_file_path = namedtuple('Output',['OutputPath_Name'])
        return upload_file_path(model)

    
    # Training a Transformer-based Session-based Recommendation Model

    import os
    import glob

    import torch 
    import transformers4rec.torch as tr

    from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt
    from transformers4rec.torch.utils.examples_utils import wipe_memory

    # As we did above, we start with defining our schema object and filtering only the `product_id` feature for training.
    from merlin_standard_lib import Schema
    # Define schema object to pass it to the TabularSequenceFeatures class
    INPUT_DATA_DIR = Data_Input

    SCHEMA_PATH = os.path.join(INPUT_DATA_DIR, 'schema_tutorial.pb')
    schema = Schema().from_proto_text(SCHEMA_PATH)
    schema = schema.select_by_name(['product_id-list_seq'])

    # Define input block
    sequence_length, d_model = 20, 192
    # Define input module to process tabular input-features and to prepare masked inputs
    inputs= tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=sequence_length,
        d_output=d_model,
        masking="mlm",
    )

    # Define XLNetConfig class and set default parameters for HF XLNet config  
    transformer_config = tr.XLNetConfig.build(
        d_model=d_model, n_head=4, n_layer=2, total_seq_length=sequence_length
    )
    # Define the model block including: inputs, masking, projection and transformer block.
    body = tr.SequentialBlock(
        inputs, tr.MLPBlock([192]), tr.TransformerBlock(transformer_config, masking=inputs.masking)
    )

    # Define the head for to next item prediction task 
    head = tr.Head(
        body,
        tr.NextItemPredictionTask(weight_tying=True, hf_format=True, 
                                metrics=[NDCGAt(top_ks=[10, 20], labels_onehot=True),  
                                        RecallAt(top_ks=[10, 20], labels_onehot=True)]),
    )

    # Get the end-to-end Model class 
    model = tr.Model(head)

    from transformers4rec.config.trainer import T4RecTrainingArguments
    from transformers4rec.torch import Trainer

    #Set arguments for training 
    training_args = T4RecTrainingArguments(
                output_dir=training_args_output_dir,
                max_sequence_length=training_args_max_sequence_length,
                data_loader_engine=training_args_data_loader_engine,
                num_train_epochs=training_args_num_train_epochs, 
                dataloader_drop_last=training_args_dataloader_drop_last,
                per_device_train_batch_size = training_args_per_device_train_batch_size,
                per_device_eval_batch_size = training_args_per_device_eval_batch_size,
                gradient_accumulation_steps = training_args_gradient_accumulation_steps,
                learning_rate=training_args_learning_rate,
                report_to = training_args_report_to,
                logging_steps=training_args_logging_steps,
            )

    # Instantiate the T4Rec Trainer, which manages training and evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        schema=schema,
        compute_metrics=True,
    )

    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", Data_Output)

    # os.system("%%time")
    start_time_window_index = 1
    final_time_window_index = 4
    for time_index in range(start_time_window_index, final_time_window_index):
        # Set data 
        time_index_train = time_index
        time_index_eval = time_index + 1
        train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))
        eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))
        # Train on day related to time_index 
        print('*'*20)
        print("Launch training for day %s are:" %time_index)
        print('*'*20 + '\n')
        trainer.train_dataset_or_path = train_paths
        trainer.reset_lr_scheduler()
        trainer.train()
        trainer.state.global_step +=1
        # Evaluate on the following day
        trainer.eval_dataset_or_path = eval_paths
        train_metrics = trainer.evaluate(metric_key_prefix='eval')
        print('*'*20)
        print("Eval results for day %s are:\t" %time_index_eval)
        print('\n' + '*'*20 + '\n')
        for key in sorted(train_metrics.keys()):
            print(" %s = %s" % (key, str(train_metrics[key]))) 
        wipe_memory()

        # Train XLNET with Side Information for Next Item Prediction

    import os
    import glob
    import nvtabular as nvt

    import torch 
    import transformers4rec.torch as tr

    from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt
    from transformers4rec.torch.utils.examples_utils import wipe_memory

    # Define categorical and continuous columns to fed to training model 
    # add it in parameters
    x_cat_names = ['product_id-list_seq', 'category_id-list_seq', 'brand-list_seq']
    x_cont_names = ['product_recency_days_log_norm-list_seq', 'et_dayofweek_sin-list_seq', 'et_dayofweek_cos-list_seq', 
                    'price_log_norm-list_seq', 'relative_price_to_avg_categ_id-list_seq']


    from merlin_standard_lib import Schema

    # Define schema object to pass it to the TabularSequenceFeatures class
    INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/dli/task/data/")

    SCHEMA_PATH = os.path.join(INPUT_DATA_DIR, 'schema_tutorial.pb')
    schema = Schema().from_proto_text(SCHEMA_PATH)
    schema = schema.select_by_name(x_cat_names + x_cont_names)

    # Define input block
    sequence_length, d_model = 20, 192
    # Define input module to process tabular input-features and to prepare masked inputs
    inputs= tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=sequence_length,
        aggregation="concat",
        d_output=d_model,
        masking="mlm",
    )

    # Define XLNetConfig class and set default parameters for HF XLNet config  
    transformer_config = tr.XLNetConfig.build(
        d_model=d_model, n_head=4, n_layer=2, total_seq_length=sequence_length
    )
    # Define the model block including: inputs, masking, projection and transformer block.
    body = tr.SequentialBlock(
        inputs, tr.MLPBlock([192]), tr.TransformerBlock(transformer_config, masking=inputs.masking)
    )

    # Define the head related to next item prediction task 
    head = tr.Head(
        body,
        tr.NextItemPredictionTask(weight_tying=True, hf_format=True, 
                                        metrics=[NDCGAt(top_ks=[10, 20], labels_onehot=True),  
                                                RecallAt(top_ks=[10, 20], labels_onehot=True)]),
    )

    # Get the end-to-end Model class 
    model = tr.Model(head)

    from transformers4rec.config.trainer import T4RecTrainingArguments
    from transformers4rec.torch import Trainer

    #Set arguments for training 
    training_args = T4RecTrainingArguments(
            output_dir=training_args_output_dir,
            max_sequence_length=training_args_max_sequence_length,
            data_loader_engine=training_args_data_loader_engine,
            num_train_epochs=training_args_num_train_epochs, 
            dataloader_drop_last=training_args_dataloader_drop_last,
            per_device_train_batch_size = training_args_per_device_train_batch_size,
            per_device_eval_batch_size = training_args_per_device_eval_batch_size,
            gradient_accumulation_steps = training_args_gradient_accumulation_steps,
            learning_rate=training_args_learning_rate,
            report_to = training_args_report_to,
            logging_steps=training_args_logging_steps,
    )

    # Instantiate the T4Rec Trainer, which manages training and evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        schema=schema,
        compute_metrics=True,
    )

    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", Data_Output)

    start_time_window_index = 1
    final_time_window_index = 4
    for time_index in range(start_time_window_index, final_time_window_index):
        # Set data 
        time_index_train = time_index
        time_index_eval = time_index + 1
        train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))
        eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))
        # Train on day related to time_index 
        print('*'*20)
        print("Launch training for day %s are:" %time_index)
        print('*'*20 + '\n')
        trainer.train_dataset_or_path = train_paths
        trainer.reset_lr_scheduler()
        trainer.train()
        trainer.state.global_step +=1
        # Evaluate on the following day
        trainer.eval_dataset_or_path = eval_paths
        train_metrics = trainer.evaluate(metric_key_prefix='eval')
        print('*'*20)
        print("Eval results for day %s are:\t" %time_index_eval)
        print('\n' + '*'*20 + '\n')
        for key in sorted(train_metrics.keys()):
            print(" %s = %s" % (key, str(train_metrics[key]))) 
        wipe_memory()

    # Exporting the preprocessing worflow and model for deployment to Triton server
    import nvtabular as nvt
    workflow_path = os.path.join(INPUT_DATA_DIR, 'workflow_etl')
    print(workflow_path)
    workflow = nvt.Workflow.load(workflow_path)

    # dictionary representing max sequence length for the sequential (list) columns
    sparse_features_max = {
        fname: sequence_length
        for fname in x_cat_names + x_cont_names
    }

    sparse_features_max


    model_path="/dli/task/model_repository"
    name= "t4r_pytorch"
    from nvtabular.inference.triton import export_pytorch_ensemble
    export_pytorch_ensemble(
        model,
        workflow,
        sparse_max=sparse_features_max,
        name= Output_Model_Name,
        model_path= Output_Model_Path,
        label_columns =[],
    )

    #implement named tuple
    from collections import namedtuple
    model_info = namedtuple('Model',['Model_Path','Model_Name'])
    return model_info(Output_Model_Path,Output_Model_Name)
