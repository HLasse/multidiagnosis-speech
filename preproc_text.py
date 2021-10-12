import pandas as pd
from pathlib import Path

# PREPROC NEEDED
# Strip (*) and blanks at the end
# Remoce [sic]
# Remove ((mumler)) and (griner)
# Vowel transcription is not uniform

# QUESTIONS
# What's the deal with missing ids (give Riccardo ids)
# What is Task NaN? Should it be excluded?
# What is Trial? Does it matter? Is it each bit of the story?
# Recovered, does it matter?

# NOTES
# Diagnosis2 has many levels
# Explain studies
# Difference between ID and SharedID

# NEXT
# Modularize this script
# Convert diagnosis to multi-class
# Get train val test splits

if __name__=='__main__':
    # Load  
    fpath = Path('data') / 'transcripts'
    df = pd.read_csv(fpath/'participants.csv',  sep=';')

    # Removing 3 rows
    df = df[~((df['Gender ']=='F/M')|(df['Age'].isnull()))]

    # Only storing relevant columns
    df.drop(['SharedID', 
            'OverallStudy', 
             'Diagnosis2', 
             'Education', 
             'Unnamed: 11'], axis=1, inplace=True)


    df['ID'] = df['ID'].astype(str)
    df.to_csv(fpath/'processed'/'participants.tsv', 
            sep='\t')

    # Read in transcripts
    trans = pd.read_csv(fpath/'data.csv')

    no_trans = set(trans.Subject.unique()) - set(df['ID'].unique())
    print(f'Transcript absent for {len(no_trans)} participants, excluding from transcript file...')
    trans = trans[~trans['Subject'].isin(no_trans)]

    no_id = set(df['ID'].unique()) - set(trans.Subject.unique())
    print(f'Diagnosis absent for {len(no_id)} participants, excluding from transcript file...')
    trans = trans[~trans['Subject'].isin(no_id)]
    # This step ends up excluding 47 participants...

    # Print excluded info
    print(f'No transcript: {no_trans}')
    print(f'No participant information: {no_id}')

    # Dropping irrelevant columns
    trans.drop(['Start Time', 'End Time', 
                'Transcriber', 'File', 'Sub File'],
               axis=1, inplace=True)

    # Log and save
    print(f'We have {trans.shape} transcripts...')

    trans.to_csv(fpath/'processed'/'transcripts.tsv', 
                 sep='\t')

    



