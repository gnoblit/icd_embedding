def main(
        data_path = '/home/gnoblit/takehome/codametrix/data/clean/backup_pairwise_train.parquet',
        model_path: str='/home/gnoblit/takehome/codametrix/models/',
        model: str='sentence-transformers/all-MiniLM-L12-v2',
):

    import polars as pl
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses import CosineSimilarityLoss, CoSENTLoss
    from datasets import load_dataset
    from datetime import datetime
    import logging
    import torch

    output_dir = model_path + model.split('/')[-1] + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f'Output dir: {output_dir}')

    model = SentenceTransformer(model, trust_remote_code=True))
    print(f'old number of tokens {len(model.tokenizer)}')
    # Add codes as tokens
    code = pl.read_ndjson('/home/gnoblit/takehome/codametrix/data/clean/raw_icd10.ndjson')
    codes = code['code'].to_list()
    word_embedding_model = model._first_module()   #Your models.Transformer object
    word_embedding_model.tokenizer.add_tokens(codes, special_tokens=False)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
    print(f'new number of tokens {len(model.tokenizer)}')

    loss = CosineSimilarityLoss(model=model, )

    dataset = load_dataset(data_files='triplet_data.parquet', path=data_path, split='train').remove_columns('genre')    

    del code
    del codes
    del df_

    print('train data')
    logging.info(dataset)


if __name__ == "__init__":
    main()