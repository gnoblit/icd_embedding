
def main(
        pos_path: str='/home/gnoblit/takehome/codametrix/data/clean/positive_train_data.ndjson', 
        neg_path: str='/home/gnoblit/takehome/codametrix/data/clean/negative_train_data.ndjson', 
        write_path: str='/home/gnoblit/takehome/codametrix/data/clean/',
        model: str='sentence-transformers/all-MiniLM-L12-v2'):
    import polars as pl
    from sentence_transformers import SentenceTransformer
    from scipy.spatial.distance import cosine
    from alive_progress import alive_bar

    positives_df = pl.read_ndjson(pos_path).drop('positive').rename(
        {
            'description': 'description_right', 
            'description_right': 'description_anchor',
            'code_right': 'code_anchor',
            'code': 'code_right'
        }
    ).select(['code_anchor', 'description_anchor', 'code_right', 'description_right']).sort('code_anchor')
    print(f'positives read, size: {positives_df.shape}')
    negatives_df = pl.read_ndjson(neg_path).drop(['positive']).rename(
        {
            'code': 'code_anchor',    
            'description': 'description_anchor'
        }
    )
    print(f'negative read, size: {negatives_df.shape}')
    
    df = pl.concat([positives_df, negatives_df])
    print('dfs concatenated')
    # Generate cosine similarity between labels. Use this to train 
    model = SentenceTransformer(model, trust_remote_code=True)
    print('model loaded')
    embeddings_1 = model.encode(df['description_anchor'].to_list())
    print('first column embedded')
    embeddings_2 = model.encode(df['description_right'].to_list())
    print('second column embedded')
    distances = []
    with alive_bar(len(embeddings_2)) as bar:
        for i_, j_ in zip(embeddings_1, embeddings_2):
            cosine_sim = 1-cosine(i_, j_)
            distances.append(cosine_sim)
            bar()

    df = df.with_columns(
        cosine_sim = distances
    )
    del embeddings_1
    del embeddings_2
    del model

    print('done with cosine')
    # Want to train on following pairs: text-text, id-id, id-text
    train_df = pl.concat(
        [
            df.select(['code', 'code_right', 'cosine_sim']).rename({'code': 'anchor', 'code_right': 'other'}),
            df.select(['code', 'description_right', 'cosine_sim']).rename({'code': 'anchor', 'description_right': 'other'}),
            #df.select(['description_anchor', 'code_right', 'cosine_sim']).rename({'description_anchor': 'anchor', 'code_right': 'other'}),
            df.select(['description_anchor', 'description_right', 'cosine_sim']).rename({'description_anchor': 'anchor', 'description_right': 'other'})
        ]
    )
    del df
    print(f'training df size: {train_df.shape}')
    train_df = train_df.sample(fraction=1, shuffle=True)

    print(f'Size of training data: {train_df.shape}')

    print(train_df.head())

    train_df.write_ndjson(write_path + 'backup_pairwise_train.ndjson')
    train_df.write_parquet(write_path + 'backup_pairwise_train.parquet')
    
if __name__ == "__main__":
    main()