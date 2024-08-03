def main(data_path = '/home/gnoblit/takehome/codametrix/data/clean/',
        model_path: str='/home/gnoblit/takehome/codametrix/models/',
        model: str='sentence-transformers/all-MiniLM-L12-v2',
        tokenize: bool=True,
        ):

    import polars as pl
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformerTrainer
    from sentence_transformers.losses import CosineSimilarityLoss, CoSENTLoss
    from datasets import load_dataset
    from datetime import datetime
    import logging
    import torch

    output_dir = model_path + model.split('/')[-1] + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f'Output dir: {output_dir}')

    model = SentenceTransformer(model, trust_remote_code=True)
    if tokenize:
            
        print(f'old number of tokens {len(model.tokenizer)}')
        # Add codes as tokens
        code = pl.read_ndjson('/home/gnoblit/takehome/codametrix/data/clean/raw_icd10.ndjson')
        codes = code['code'].to_list()
        word_embedding_model = model._first_module()   #Your models.Transformer object
        word_embedding_model.tokenizer.add_tokens(codes, special_tokens=False)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        print(f'new number of tokens {len(model.tokenizer)}')
        del code
        del codes

    loss = CosineSimilarityLoss(model=model, )

    dataset = load_dataset(data_files='backup_pairwise_train.parquet', path=data_path, split='train')

    print('train data')
    logging.info(dataset)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=16,
        # per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.0,
        auto_find_batch_size=True,
        # eval_strategy="steps",
        # eval_steps=.05,
        save_strategy="steps",
        save_steps=.05,
        logging_strategy='steps',
        logging_steps=.05,
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

if __name__ == "__main__":
    main()