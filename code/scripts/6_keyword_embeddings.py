def main(model_path: str='/home/gnoblit/takehome/codametrix/models/all-MiniLM-L6-v2_2024-08-02_20-10-13/checkpoint-197000',
         top_k: int=5,
         ):
    """Function exists to generate embeddings for all words within the corpus of descriptions associated with the ICD10 descriptions."""

    import polars as pl
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import semantic_search
    import torch

    corpus = pl.read_ndjson('/home/gnoblit/takehome/codametrix/data/clean/raw_icd10.ndjson')
    codes = corpus['code'].to_list()
    descriptions = corpus['description'].to_list()
    

    descriptions = corpus['description'].str.split(' ').to_list()

    s = set()
    descriptions = corpus['description'].str.split(' ').to_list()
    for i_ in descriptions:
        s.update(set(i_))
    description_words = list(set([''.join(filter(str.isalpha, el_)).lower() for el_ in s if len(''.join(filter(str.isalnum, el_)))>3]))
    print(f'Description words: {description_words[:10]}')

    del corpus 
    
    print(f'Number of unique words in descriptions: {len(description_words)}')

    model_hex = model_path.split('/')[-2]
    model = SentenceTransformer(model_path)

    # Embed words
    word_embeddings = model.encode(description_words)
    code_embeddings = model.encode(codes)

    # Save embeddings
    word_df = pl.DataFrame({'words':description_words, 'embedding':word_embeddings})
    print(word_df.head())
    code_df = pl.DataFrame({'codes':codes, 'embeddings':code_embeddings})
    print(code_df.head())
    print(code_df.tail())

    

    word_df.write_ndjson(f'{model_path}/words.ndjson')
    code_df.write_ndjson(f'{model_path}/codes.ndjson')

    hits = semantic_search(code_embeddings, word_embeddings, top_k=top_k)

    # Write hits
    solution = []
    for iter_, q_ in enumerate(hits):
            keywords = []
            scores = []
            for i_ in q_:
                 keywords.append(description_words[i_['corpus_id']])
                 scores.append(i_['score'])
            solution.append(
                {
                      'code': codes[iter_],
                      'keywords': keywords,
                      'scores': scores,
                      'keyword_1': keywords[0],
                      'keyword_2': keywords[1],
                      'keyword_3': keywords[2],
                      'keyword_4': keywords[3],
                      'keyword_5': keywords[4],
                      'score_1': scores[0],
                      'score_2': scores[1],
                      'score_3': scores[2],
                      'score_4': scores[3],
                      'score_5': scores[4],
                      
                }
            )

    solution_df = pl.from_dicts(solution)
    solution_df = solution_df.sort('code')
    print(solution_df.sample(10).select(['code', 'keywords', 'scores']))
    solution_df.write_ndjson(f'{model_path}/keyword_solutions.ndjson')
    

def flatten(arg) -> list:
    """Helper function to recursively flatten a list of lists"""
    if not isinstance(arg, list): # if not list
        return [arg.lower()]
    return [x for sub in arg for x in flatten(sub)]


if __name__ == "__main__":
    main()