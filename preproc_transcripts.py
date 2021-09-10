import pandas as pd
from pathlib import Path


RAW_PATH = Path('data') / 'raw'
PROCESSED_PATH = Path('data') / 'processed'


# Read in trial data, get participant descriptors, only keep non acoustic features
def preprocess():
    by_trial = pd.read_csv(RAW_PATH / 'labels.csv')
    outs = ['Diagnosis',
            'Gender',
            'Age',
            'AdosCommunication',
            'AdosSocial',
            'AdosCreativity',
            'AdosStereotyped',
            'VIQ',
            'PIQ',
            'TIQ',
            'ParentalEducation',
            'SRS',
            'CARS',
            'PPVT',
            'Leiter',
            'language',
            'AgeS']
    labels = by_trial.groupby('ID')[outs].first().reset_index()

    # Remove missing IDs
    stories = pd.read_csv(RAW_PATH / 'stories.tsv', sep='\t')
    stories['ID'] = stories['Subject'].str.rstrip('A')
    triangles = pd.read_csv(RAW_PATH / 'triangles.tsv', sep='\t')
    for r in triangles[triangles['Subject'].isnull()].index.tolist():
        triangles.loc[r, 'Subject'] = triangles.loc[r-1, 'Subject']
    triangles = triangles[triangles['Subject']!='Malthe']
    triangles['ID'] = triangles['Subject'].str.rstrip('A')

    # Process stories and make sure all ids are in place
    stories_no_id = set(stories.ID.unique()) - set(labels.ID.unique())
    replace_dict = {'330':'230', '331': '231'}
    stories['ID'] = stories['ID'].replace(replace_dict)
    dk_no_story = labels[(~labels['ID'].isin(stories.ID.unique())) & 
                        (labels['language']=='dk')].ID.tolist()
    stories_no_id_new = set(stories.ID.unique()) - set(labels.ID.unique())
    assert stories_no_id_new == set()

    # Process triangles and make sure all ids are in place
    triangles_no_id = set(triangles.ID.unique()) - set(labels.ID.unique())
    assert triangles_no_id == set()
    dk_no_triangles = labels[(~labels['ID'].isin(triangles.ID.unique())) & 
                            (labels['language']=='dk')].ID.tolist()
    # Note that we are missing a few Ados scores (but Diagnosis is defined everywhere)

    # Log stuff
    print(f'There are dk {len(dk_no_story)} participant with no story data')
    print(f'There are {len(dk_no_triangles)} participant with no triangle data')

    # Keep relevant columns
    cols = ['ID', 'File', 'Task', 'Transcript']
    stories = stories[cols]
    triangles = triangles[cols]
    stories['language'] = 'dk'
    triangles['language'] = 'dk'

    # Store
    print('Saving...')
    labels.to_csv(PROCESSED_PATH / 'labels.tsv', sep='\t')
    stories.to_csv(PROCESSED_PATH / 'stories.tsv', sep='\t')
    triangles.to_csv(PROCESSED_PATH / 'triangles.tsv', sep='\t')

if __name__=='__main__':
    preprocess()