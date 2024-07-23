def clean_data(text_path: str='/home/gnoblit/takehome/codametrix/data/supplied/icd10cm_order_2021.txt', write_path: str='/home/gnoblit/takehome/codametrix/data/clean'):
    import ndjson
    import json
    import polars as pl

    lines = []
    tree = {}

    with open(text_path) as f:
        for line_ in f:
            path = f'{line_[6]}-{line_[6:9].rstrip()}-{line_[6:10].rstrip()}-{line_[6:11].rstrip()}-{line_[6:12].rstrip()}-{line_[6:13].rstrip()}'
            path = create_path(path)
            path_split = path.split('-')
            ancestors = '-'.join(path.split('-')[1:-1])

            file = {
                'path': path,
                'ancestors': ancestors,
                'code': line_[6:14].rstrip(), 
                'category': line_[6:9].rstrip(),
                'details': line_[9:13].rstrip(),

                'section': line_[6].rstrip() if line_[6].rstrip() else 'None',
                'part': line_[7].rstrip() if line_[7].rstrip() else 'None',
                'root_operation': line_[8].rstrip() if line_[8].rstrip() else 'None',

                'etiology': line_[9].rstrip() if line_[9].rstrip() else 'None',
                'location': line_[10].rstrip() if line_[10].rstrip() else 'None',
                'laterality': line_[11].rstrip() if line_[11].rstrip() else 'None',
                
                'extension': line_[12].rstrip() if line_[12].rstrip() else 'None',

                'up_to_etiology': path_split[2] if len(path_split)>2 else "None",

                'up_to_location': path_split[3] if len(path_split)>3 else "None",
                'up_to_laterality': path_split[4] if len(path_split)>4 else "None",

                'description': line_.rstrip()[77:]
            } 
            lines.append(file)

            nodes = file['path'].split('-')
            for iter_ in range(len(nodes)-1):
                tree.setdefault(nodes[iter_], set()).add(nodes[iter_+1])
    # Saves    
    with open(write_path + '/raw_icd10.ndjson', 'w') as f:
        ndjson.dump(lines, f)
    with open(write_path + '/icd10_tree.json', 'w') as f:
        json.dump(tree, f, default=set_default)

def set_default(obj: set) -> list:
    """Function exists to convert set to list for json dump. Returns sorted list."""
    if isinstance(obj, set):
        to_return = list(obj)
        to_return.sort()
        return to_return
    raise TypeError

def create_path(path: str):
    """Function takes in string of path and eliminates any repeated nodes."""
    path = path.split('-')
    r = 1

    while r < len(path):
        if path[r] == path[r - 1]:
            break
        r += 1
    return '-'.join(path[:r])

if __name__ == '__main__':
    clean_data()