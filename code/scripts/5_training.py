import polars as pl
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import TripletLoss

from datasets import load_dataset, DatasetDict

from datetime import datetime
import logging

def main(
        triplets_path: str='/home/gnoblit/takehome/codametrix/data/clean/',
        model: str='sentence-transformers/all-mpnet-base-v2',
        model_path: str='/home/gnoblit/takehome/codametrix/models/',
        ):
    """
    Function exists to train model using triplets data
    """
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


    output_dir = model_path + model.split('/')[-1] + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f'Output dir: {output_dir}')

    model = SentenceTransformer(model)
    loss = TripletLoss(model=model)

    dataset = load_dataset(data_files=triplets_path + 'triplet_data.parquet', path=triplets_path, num_proc=-1, split='train[:10000]' )
    dataset = dataset.shuffle().remove_columns('genre')
    print('dataset: ', dataset)
    

    train_test_split  = dataset.train_test_split(test_size=0.1)
    test_valid = train_test_split['test'].train_test_split(test_size=0.5)
    train_test_valid_datasets = DatasetDict({
        'train': train_test_split['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})
    
    del dataset

    logging.info(train_test_valid_datasets['train'])

    dev_evaluator = TripletEvaluator(
        anchors=train_test_valid_datasets['valid']["anchor"],
        positives=train_test_valid_datasets['valid']["positive"],
        negatives=train_test_valid_datasets['valid']["negative"],
        name="triplet_valid",
        show_progress_bar=True
    )
    dev_evaluator(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        auto_find_batch_size=True,
        eval_strategy="steps",
        eval_steps=.05,
        save_strategy="steps",
        save_steps=.25,
        logging_strategy='steps',
        logging_steps=.05,
        logging_first_step=True,
        save_total_limit=4,
        run_name=output_dir.split('/')[-1],  
        report_to='wandb'
    )
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_test_valid_datasets['train'],
        loss=loss,
        eval_dataset=train_test_valid_datasets['valid'],
        evaluator=dev_evaluator
    )
    trainer.train()

    model.save_pretrained(output_dir + '/fine_tuned')

    test_evaluator = TripletEvaluator(
        anchors=train_test_valid_datasets['test']["anchor"],
        positives=train_test_valid_datasets['test']["positive"],
        negatives=train_test_valid_datasets['test']["negative"],
        name="triplet_test",
        show_progress_bar=True
    )

    test_evaluator(model)

if __name__ == '__main__':
    main()