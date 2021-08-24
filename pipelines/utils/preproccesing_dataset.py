import json
import os
from pathlib import Path

def preprocces_metadata(root_path):
    """
    takes root folder for dataset and gives 2 .txt files with paths for all images
    """
    data = {}
    data['dbImage'] = []
    data['qImage'] = []
    exts = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
    paths = []
    for ext in exts:
        paths += list(Path(root_path).glob('**/'+ext))
    db_image_path = 'database_images_path.txt'
    query_image_path = 'query_images_path.txt'
    qcount = dbcount = 0
    with open(db_image_path, 'w') as dbfile, \
         open(query_image_path, 'w') as qfile:
            for filename in sorted(paths):
                if str(filename).find('.png') == -1:
                    continue
                if (str(filename).find('database') or str(filename).find('mapping')) != -1:
                    dbcount += 1
                    dbfile.write(os.path.join(filename) + '\n')
                if str(filename).find('query') != -1:
                    qcount += 1
                    qfile.write(os.path.join(filename) +'\n')

    print(f'Was loaded {qcount} query and {dbcount} database images')
    return query_image_path, db_image_path