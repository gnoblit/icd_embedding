from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss

from datasets import load_dataset, load_from_disk, DatasetDict

from datetime import datetime
import logging

import polars as pl

def main(
        triplets_path: str='/home/gnoblit/takehome/codametrix/data/clean/',
        model: str='sentence-transformers/all-MiniLM-L6-v2',
        model_path: str='/home/gnoblit/takehome/codametrix/models/',
        ):
    """
    Function exists to train model using triplets data
    """
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


    output_dir = model_path + model.split('/')[-1] + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f'Output dir: {output_dir}')
    code = pl.read_ndjson('/home/gnoblit/takehome/codametrix/data/clean/raw_icd10.ndjson')
    codes = code['code'].to_list()

    model = SentenceTransformer(model, trust_remote_code=True)
    print(f'old number of tokens {len(model.tokenizer)}')
    # Add codes as tokens
    word_embedding_model = model._first_module()   #Your models.Transformer object
    word_embedding_model.tokenizer.add_tokens(codes, special_tokens=False)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
    print(f'new number of tokens {len(model.tokenizer)}')

    loss = MultipleNegativesRankingLoss(model=model)

    dataset = load_dataset(data_files='triplet_data.parquet', path=triplets_path, split='train').remove_columns('genre')

    del code
    del codes
    # train_test_split = dataset.train_test_split(test_size=0.001)
    # test_valid = train_test_split['test'].train_test_split(test_size=0.5)
    # train_test_valid_datasets = DatasetDict({
    #     'train': train_test_split['train'],
    #     #'test': test_valid['test'],
    #     #'valid': test_valid['train']
    #     })
    
    print('train data')
    logging.info(dataset)

    # dev_evaluator = TripletEvaluator(
    #     anchors=train_test_valid_datasets['valid']["anchor"],
    #     positives=train_test_valid_datasets['valid']["positive"],
    #     negatives=train_test_valid_datasets['valid']["negative"],
    #     name="triplet_valid",
    #     show_progress_bar=True
    # )
    # dev_evaluator(model)
    # print('dev_evaluator')
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=64,
        # per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.0,
        auto_find_batch_size=True,
        # eval_strategy="steps",
        # eval_steps=.05,
        save_strategy="steps",
        save_steps=.05,
        logging_strategy='steps',
        logging_steps=.01,
        logging_first_step=True,
        run_name=output_dir.split('/')[-1],  
        report_to='wandb'
    )
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
        # eval_dataset=train_test_valid_datasets['valid'],
        # evaluator=dev_evaluator
    )
    print('about to train')
    trainer.train()

    model.save_pretrained(output_dir + '/fine_tuned')

    # test_evaluator = TripletEvaluator(
    #     anchors=train_test_valid_datasets['test']["anchor"],
    #     positives=train_test_valid_datasets['test']["positive"],
    #     negatives=train_test_valid_datasets['test']["negative"],
    #     name="triplet_test",
    #     show_progress_bar=True
    # )

    # test_evaluator(model)

if __name__ == '__main__':
    main()