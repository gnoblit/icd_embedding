
import polars as pl

import logging

def main(
        pos_path: str='/home/gnoblit/takehome/codametrix/data/clean/positive_train_data.ndjson', 
        neg_path: str='/home/gnoblit/takehome/codametrix/data/clean/negative_train_data.ndjson', 
        data_path: str='/home/gnoblit/takehome/codametrix/data/clean/raw_icd10.ndjson',
        write_path: str='/home/gnoblit/takehome/codametrix/data/clean/',
        include_codes: bool=True):

    d = pl.read_ndjson(data_path)
    d = d.with_columns(pl.col('description').str.split(' ').alias('description')).select(['code', 'description']).explode('description').sort('code').select(['code', 'description'])
    d = d.with_columns(label=pl.lit(1))
    print(d.head())
    
    # positives_df = pl.read_ndjson(pos_path).rename(
    #     {
    #         'description': 'description_right', 
    #         'description_right': 'description_anchor',
    #         'code_right': 'code_anchor',
    #         'code': 'code_right',
    #         'positive': 'label'
    #     }
    # ).select(['code_anchor', 'description_right', 'label']).sort('code_anchor')
    # positives_df

    # print(f'positives read, size: {positives_df.shape}')
    # print(positives_df.head())

    # negatives_df = pl.read_ndjson(neg_path).rename(
    #     {
    #         'code': 'code_anchor',    
    #         'description': 'description_anchor',
    #         'positive': 'label'
    #     }
    # )
    # print(f'negative read, size: {negatives_df.shape}')

    negatives_df = pl.read_ndjson(neg_path).rename(
        {
            'positive': 'label'
        }
    ).select(['code', 'description_right', 'label']).rename({'description_right':'description'})

    negatives_df = negatives_df.with_columns(
        pl.col('description').str.split(' ').alias('description')
    ).explode('description').sort('code')
    negatives_df = negatives_df.with_columns(
        label=pl.lit(0)
    )

    print(negatives_df.head())

    if include_codes:
        positive_codes = pl.read_ndjson(pos_path).select(['code', 'code_right']).rename(
            {
                'code_right': 'code',
                'code': 'code_right',
            }
        ).select(['code', 'code_right']).rename({'code_right': 'description'}).with_columns(label=pl.lit(1)).sort('code')

        negative_codes = pl.read_ndjson(neg_path).select(['code', 'code_right']).rename({'code_right':'description'}).with_columns(label=pl.lit(0))

        print(d.shape, negatives_df.shape, positive_codes.shape, negative_codes.shape)
        df = pl.concat([d, negatives_df, positive_codes, negative_codes])
        del d
        del negative_codes
        del negatives_df
        del positive_codes

    else:
        print(d.shape, negatives_df.shape)
        df = pl.concat([d, negatives_df])
        del d
        del negatives_df
    
    # train_df = pl.concat(
    #     [
    #         df.select(['code_anchor', 'code_right', 'label']).rename({'code_anchor': 'sentence1', 'code_right': 'sentence2'}),
    #         df.select(['code_anchor', 'description_right', 'label']).rename({'code_anchor': 'sentence1', 'description_right': 'sentence2'}),
    #         df.select(['description_anchor', 'code_right', 'label']).rename({'description_anchor': 'sentence1', 'code_right': 'sentence2'}),
    #     ]
    # )

    # del df
    train_df = df.sample(fraction=1, shuffle=True).rename({
        'code': 'sentence1',
        'description': 'sentence2',
    })
    del df
    train_df = train_df.with_columns(
        pl.col('sentence2').str.replace('[^a-zA-Z]', '')
    )

    print(f'Size of training data: {train_df.shape}')

    print(train_df.head())

    train_df.write_ndjson(write_path + 'backup_pairwise_train.ndjson')
    train_df.write_parquet(write_path + 'backup_pairwise_train.parquet')
    
if __name__ == "__main__":
    main()