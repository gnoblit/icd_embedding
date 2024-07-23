import polars as pl
from alive_progress import alive_bar

def generate_data(data_path: str='/home/gnoblit/takehome/codametrix/data/clean/raw_icd10.ndjson', write_path: str='/home/gnoblit/takehome/codametrix/data/clean/', n: int=5):
    df = pl.read_ndjson(data_path)

    negatives = []

    negatives = generate_negatives(df, df, 45)
    negative_df = pl.concat(negatives)
    print(f'Done with negatives, shape is: {negative_df.shape}')
    print(negative_df.head())
    negative_df.write_ndjson(write_path + 'negative_train_data.ndjson')

def generate_negatives(df, master_df, n: int):
    """
    Subset negatives not section.
    Randomly draw n
    Label as negatives
    """
    negatives = []

    with alive_bar(df.shape[0]) as bar:
            
        for iter_, el_ in enumerate(df.iter_rows(named=True)):
            subset_df = master_df.filter(
                pl.col('section') != el_['code'][:1]
            ).sample(n)
        
            subset_df = subset_df.select(['code', 'section', 'description'])
            subset_df = subset_df.rename(
                {
                    'code':'code_right',
                    'description': 'description_right'
                }
            )
            subset_df = subset_df.with_columns(
                positive=pl.lit(False),
                code=pl.lit(el_['code']),
                description=pl.lit(el_['description'])
            )
            
            # subset_df = subset_df.with_columns(
            #     codes = pl.concat_list('code', 'code_right')
            # )
            # subset_df = subset_df.with_columns(
            #     codes = pl.col('codes').list.sort()
            # )

            subset_df = subset_df.select(
                [
                    'code',
                    'description',
                    'code_right',
                    'description_right',
                    # 'codes',
                    'positive'
                ]
            )
            
            negatives.append(subset_df)
            bar()
    return negatives

if __name__ == '__main__':
    generate_data()