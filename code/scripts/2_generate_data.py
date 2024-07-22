"""Script exists to generate dataset. Will use previously provided tree structure to generate the nearby cases that the model will mine for similar codes and descriptions. Will consider codes and descriptions from other parts of the tree to be negative cases."""
import polars as pl

def generate_data(data_path: str='/home/gnoblit/takehome/codametrix/data/clean/raw_icd10.ndjson', write_path: str='/home/gnoblit/takehome/codametrix/data/clean/'):
    """
    Function takes in a path to data and to the tree as well as n, an integer denoting how many negative cases to pair with. 

    Then permutes the data to create a dataset for training a sentence encoder.
    First, pairs each code with its description.
    Then, pairs each code with each ancestor (upstream node, parent node, parent of parent, etc.). These are the positive cases. 

    Negative cases are drawn up as random samples from different segments of the tree, i.e. branches that begin with different letters. 
    """

    
    df = pl.read_ndjson(data_path)
    #print(df.head())

    df = (df.with_columns(
        pl.when(
        pl.col('up_to_laterality').eq(pl.col('up_to_location'))
            )
        .then(pl.lit(''))
        .otherwise(pl.col('up_to_laterality'))
        .alias('up_to_laterality')
    ))
    
    df = (df.with_columns(
        pl.when(
            pl.col('up_to_location').eq(pl.col('up_to_etiology'))
            )
        .then(pl.lit(''))
        .otherwise(pl.col('up_to_location'))
        .alias('up_to_location'),
    ))


    # df = df.with_columns(
    #     pl.col('path').str.split('-').alias('path_list')
    # )

    # df.write_ndjson(write_path + 'train_data.ndjson')

    df = df.with_columns(
        pl.all().replace({'':None})
    )   
    # Generate training data
    dfs = []
    print('starting')
    laterality_cols = ['code', 'up_to_laterality', 'description']
    positives = join_dfs(df, laterality_cols, 'up_to_laterality')
    dfs.append(positives)
    print('done with laterality')

    location_cols = ['code', 'up_to_location', 'description']
    positives = join_dfs(df, location_cols, 'up_to_location')
    dfs.append(positives)
    print('done with location')

    etiology_cols = ['code', 'up_to_etiology', 'description']
    positives = join_dfs(df, etiology_cols, 'up_to_etiology')
    dfs.append(positives)
    print('done with etiology')

    laterality_cols = ['code', 'up_to_laterality', 'description']
    positives = join_dfs(df, laterality_cols, 'up_to_laterality')
    dfs.append(positives)
    print('done with laterality')

    category_cols = ['code', 'category', 'description']

    # dfs.append(join_dfs(df, category_cols, 'category'))
    # print('done with categories')    

    train_df = pl.concat(dfs)
    
    print(train_df.head())

    train_df.write_ndjson(write_path + 'positive_train_data.ndjson')

    # Generate negatives


def join_dfs(df, cols: list, join_term: str):
    """Function joins df subsets of cols columns on the join_term and returns unique rows"""

    df = df.select(cols).join(
        df.select(cols),
        how='left',
        on=join_term
    ).filter(~(pl.col('description').eq(pl.col('description_right')))).drop(join_term)

    df = df.with_columns(
        codes = pl.concat_list('code', 'code_right')
    )
    df = df.with_columns(
        codes = pl.col('codes').list.sort()
    )

    df = df.with_columns(
        # section=pl.col('code').str.slice(0, 1),
        positive=pl.lit(True)
    )

    df = df.unique('codes').sort('code')

    return df


if __name__ == '__main__':
    generate_data()