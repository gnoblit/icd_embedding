
import polars as pl

def main(
        pos_path: str='/home/gnoblit/takehome/codametrix/data/clean/positive_train_data.ndjson', 
        neg_path: str='/home/gnoblit/takehome/codametrix/data/clean/negative_train_data.ndjson', 
        write_path: str='/home/gnoblit/takehome/codametrix/data/clean/'):
    """Function exists to format training data for triplet loss function. From SentenceTransformers: "Given a triplet of (anchor, positive, negative), the loss minimizes the distance between anchor and positive while it maximizes the distance between anchor and negative." 

    Takes in path (str) to positive and negative training examples
    Merges negatives onto positives to construct a larger dataframe that consists of combinations of each negative file with all magine positive 
    """

    positives_df = pl.read_ndjson(pos_path).drop('positive').rename(
        {
            'description': 'description_positive', 
            'description_right': 'description_anchor',
            'code_right': 'code_anchor',
            'code': 'code_positive'
        }
    ).select(['code_anchor', 'description_anchor', 'code_positive', 'description_positive']).sort('code_anchor')
    
    negatives_df = pl.read_ndjson(neg_path).drop(['positive', 'description']).rename({'code': 'code_anchor',          
        'code_right': 'code_negative', 'description_right': 'description_negative'})
    
    train_df = positives_df.join(
        negatives_df,
        how='left',
        on='code_anchor'
        ).sort('code_anchor')
    

    print(f'Size of merged dataset: {train_df.shape}')

    print(train_df.head(10))
    print(train_df.tail(10))

    # Create subdatasets of only labels, label and codes, only codes

    only_codes = train_df.select(['code_anchor', 'code_positive', 'code_negative']).with_columns(genre=pl.lit('ccc')).rename({'code_anchor': 'anchor', 'code_positive': 'positive', 'code_negative':'negative'}).sample(fraction=1, shuffle=True)
    only_descriptions = train_df.select(['description_anchor', 'description_positive', 'description_negative']).with_columns(genre=pl.lit('ddd')).rename({'description_anchor': 'anchor', 'description_positive': 'positive', 'description_negative':'negative'})
    mix_1 = train_df.select(['code_anchor', 'description_positive', 'description_negative']).with_columns(genre=pl.lit('cdd')).rename({'code_anchor': 'anchor', 'description_positive': 'positive', 'description_negative':'negative'})
    mix_2 = train_df.select(['description_anchor', 'code_positive', 'description_negative']).with_columns(genre=pl.lit('dcd')).rename({'description_anchor': 'anchor', 'code_positive': 'positive', 'description_negative':'negative'})
    mix_3 = train_df.select(['description_anchor', 'description_positive', 'code_negative']).with_columns(genre=pl.lit('ddc')).rename({'description_anchor': 'anchor', 'description_positive': 'positive', 'code_negative':'negative'})


    train_df = pl.concat([only_codes, only_descriptions, mix_1])
    del only_codes
    del only_descriptions
    del mix_1
    del mix_2
    del mix_3

    train_df = train_df.sample(fraction=1, shuffle=True)

    print(f'final df shape: {train_df.shape}')
    print(train_df.head())
    train_df.write_parquet(write_path + 'triplet_data.parquet')
    print('done with parquet')
    train_df.write_ndjson(write_path + 'triplet_data.ndjson')
    print('json done!')
    train_df.write_csv(write_path + 'triplet_data.csv')
    print('done!')

if __name__ == '__main__':
    main()

