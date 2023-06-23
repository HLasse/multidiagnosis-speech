import pandas as pd
from pathlib import Path
import seaborn as sns

# PREPROC NEEDED
# Strip (*) and blanks at the end
# Remove [sic]
# Remove ((mumler)) and (griner)
# Vowel transcription is not uniform

# QUESTIONS
# Missing IDS
# What is Task NaN?
# What is Trial? Does it matter?
# Recovered, does it matter?

# NOTES
# Diagnosis2 has many levels
# Explain studies
# Difference between ID and SharedID

# NEXT
# Modularize this script
# Convert diagnosis to multi-class?
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


    df['ID'] = df['ID']# .astype(str)
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
    # Lasse: 0 excluded when I run it (using CleanData.csv as participants.csv)

    # Print excluded info
    print(f'No transcript: {no_trans}')
    # {448, 321, 322, 325, 327, 328, 42, 44, 303, 308, 309, 312, 313}
    print(f'No participant information: {no_id}')

    # Dropping irrelevant columns
    trans.drop(['Start Time', 'End Time', 
                'Transcriber'],
               axis=1, inplace=True)

    # Log and save
    trans = trans[trans['Group']!='Mixed']
    trans = trans[trans['Diagnosis']!='Mixed']
    trans = trans[trans['Diagnosis']!='ASD&Schizophrenia']
    print(f'We have {trans.shape[0]} transcripts...')

    # Remap names
    trans['Group'] = trans['Group'].replace({'Schizophrenia': 'SCHZ', 
                                             'Depression': 'DEPR'})
    trans['Diagnosis'] = trans['Diagnosis'].replace({'Schizophrenia': 'SCHZ', 
                                                     'Depression': 'DEPR', 
                                                     'Control': 'TD'})
    
    # Removing previously depressed patients (those in remission))
    remission = trans[(trans["Diagnosis"] == "DEPR") & (trans["Recovered"] == 1)]
    trans = trans.drop(remission.index)

    # Depression has overlapping Subject IDs between first-episode and chronic depression
    # If Sub File contains a 'dpc', the patient has chronic depression
    # Changing study to '2' for chronic depression to follow the naming convention
    trans.loc[trans["Sub File"].str.contains("dpc", na=False), "Study"] = 2

    # There are some extra controls that have transcripts but no audio nor metadata
    # They are probably the controls at second visit. There's no Sub File
    # information on them though, so can't know for sure. Removing them.
    trans = trans[trans["File"] != "Depression-Controls-DK-Triangles-2-Sheet1.csv"]
    
    trans['id'] = trans['Group'] + '_' + trans['Diagnosis'] + '_' + trans['Study'].astype(str) + '_' + trans['Subject'].astype(str)
    trans['Transcript'] = trans['Transcript'].astype(str)
    
    # Check length and exclude 
    trans['n_wds'] = trans.Transcript.apply(lambda x: len(x.split())).values
    print(f'Total transcripts: {trans.shape[0]}')
    print(f" Below 5 wds: {round((trans['n_wds'] < 5).sum() / trans.shape[0], 2)}, {(trans['n_wds'] < 5).sum()}")
    print(f" Below 10 wds: {round((trans['n_wds'] < 10).sum() / trans.shape[0], 2)}, {(trans['n_wds'] < 10).sum()}")
    
    # Plot histogram
    trans['Diagnosis_binary'] = trans['Diagnosis'].map({'ASD': '1', 
                                                        'DEPR': '1', 
                                                        'SCHZ': '1', 
                                                        'TD': '0'})
    sns.displot(data=trans, 
                x='n_wds', col='Group', 
                hue='Diagnosis_binary', 
                binwidth=5)

    # Check
    trans.drop(['Diagnosis_binary', 
                'Trial', 
                'Recovered'], axis=1).to_csv(fpath/'processed'/'transcripts.tsv', 
                                             sep='\t')

    



