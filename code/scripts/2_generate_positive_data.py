"""Script exists to generate dataset. Will use previously provided tree structure to generate the nearby cases that the model will mine for similar codes and descriptions. Assumes any ancestor is a positive pair or that any code with identical ancestors is a positive pair."""
import polars as pl
import numpy as np
from alive_progress import alive_bar

def generate_data(data_path: str='/home/gnoblit/takehome/codametrix/data/clean/raw_icd10.ndjson', write_path: str='/home/gnoblit/takehome/codametrix/data/clean/'):
    """
    Function takes in a path to data and to the tree as well as n, an integer denoting how many negative cases to pair with. 

    Then permutes the data to create a dataset for training a sentence encoder.
    First, pairs each code with its description.
    Then, pairs each code with each ancestor (upstream node, parent node, parent of parent, etc.). These are the positive cases. 

    Negative cases are drawn up as random samples from different segments of the tree, i.e. branches that begin with different letters. 
    """

    
    df = pl.read_ndjson(data_path)

    df = df.with_columns(
        pl.all().replace({'None':None})
    )   
    # Generate training data


    positives = []
    print('starting')

    with alive_bar(df.shape[0]) as bar:
        positives = []
        for iter_, el_ in enumerate(df.iter_rows(named=True)):
            
            ancestors = el_['ancestors']
            # Subset out any code in ancestors. This means I won't have duplicates (only go up tree)
            if ancestors:
                ancestors_split = el_['ancestors'].split('-')
                subset_df = df.filter(
                    (
                            ( 
                                (pl.col('code').is_in(ancestors_split)) | (pl.col('ancestors').eq(ancestors)) 
                                ) & (pl.col('code') != el_['code'])
                    )
                )
                
                subset_df = subset_df.select(['code', 'section', 'description'])
                subset_df = subset_df.rename(
                    {
                        'code':'code_right',
                        'description': 'description_right'
                    }
                )

                # Set as positive case
                subset_df = subset_df.with_columns(
                    positive=pl.lit(True),
                    code=pl.lit(el_['code']),
                    description=pl.lit(el_['description'])
                )

                positives.append(subset_df)
            bar()


    # category_cols = ['code', 'category', 'description']
    # positives = join_dfs(df, category_cols, 'category')
    # dfs.append(positives)
    # print(f'done with category, shape is: {positives.shape}')

    # laterality_cols = ['code', 'up_to_laterality', 'description']
    # positives = join_dfs(df, laterality_cols, 'up_to_laterality')
    # dfs.append(positives)
    # print(f'done with laterality, shape is: {positives.shape}')

    # location_cols = ['code', 'up_to_location', 'description']
    # positives = join_dfs(df, location_cols, 'up_to_location')
    # dfs.append(positives)
    # print(f'done with location, shape is: {positives.shape}')

    # etiology_cols = ['code', 'up_to_etiology', 'description']
    # positives = join_dfs(df, etiology_cols, 'up_to_etiology')
    # dfs.append(positives)
    # print(f'done with etiology, shape is: {positives.shape}')

    # category_cols = ['code', 'category', 'description']

    # dfs.append(join_dfs(df, category_cols, 'category'))
    # print('done with categories')    

    positives_df = pl.concat(positives)
    positives_df = positives_df.sort('code')
    positives_df = positives_df.select(
        ['code', 'description', 'code_right', 'description_right', 'positive']
    )
    
    print(f'Shape is: {positives_df.shape}')
    print(positives_df.head())

    positives_df.write_ndjson(write_path + 'positive_train_data.ndjson')

    # Generate negatives


def join_dfs(df, cols: list, join_term: str):
    """Function joins df subsets of cols columns on the join_term and returns unique rows"""

    pl.Config.set_streaming_chunk_size(100)
    df = df.select(cols)
    clone = df.clone()


    df = df.lazy().join(
        clone.lazy(),
        how='left',
        on=join_term
    ).collect(streaming=True)

    del clone
    
    df = df.filter(~(pl.col('description').eq(pl.col('description_right'))))
    df = df.drop(join_term)

    l = np.zeros(df.shape[0], 2)
    for iter_, el_ in enumerate(df.iter_rows(named=True)):
        l_temp = [el_['code'], el_['code_right']]
        l_temp.sort()
        l[iter_] = l_temp

    df = df.with_columns(
        codes=pl.Series(l),
        positive=pl.lit(True))
    # df = df.lazy().with_columns(
    #     codes=pl.concat_str('code', 'code_right')
    #     ).collect(streaming=True)
    # df = df.lazy().with_columns(
    #     codes=pl.col('codes').str.split('-')
    # ).collect(streaming=True)
    
    # df = df.lazy().with_columns(
    #     codes=pl.col('codes').list.sort()
    # ).collect(streaming=True)

    # df = df.with_columns(
    #     # section=pl.col('code').str.slice(0, 1),
    #     positive=pl.lit(True)
    #     )

    df = df.unique('codes')
    df = df.sort('code')

    return df


if __name__ == '__main__':
    generate_data()