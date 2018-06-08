import pandas as pd

df = pd.read_csv('data/final_df.csv')

# drop usless columns
df = df.drop('SUBJECT_ID', axis=1)
df = df.drop('HADM_ID', axis=1)
df = df.drop('ADMITTIME', axis=1)
df = df.drop('DISCHTIME', axis=1)

# convert excess types to other
# language take the top 7
lang_to_inclucde = list(df.LANGUAGE.value_counts()[:7].index)
df.loc[(df.LANGUAGE.notnull()) &
       (-df.LANGUAGE.isin(lang_to_inclucde)), 'LANGUAGE'] = 'OTHER'

# religion
relig_to_inclucde = list(df.RELIGION.value_counts()[:6].index)
df.loc[(df.RELIGION.notnull()) &
       (-df.RELIGION.isin(relig_to_inclucde)), 'RELIGION'] = 'OTHER'

# marital status consolidate and rename
df.loc[df.MARITAL_STATUS == 'LIFE PARTNER', 'MARITAL_STATUS'] = 'OTHER'
df.loc[df.MARITAL_STATUS == 'UNKNOWN (DEFAULT)', 'MARITAL_STATUS'] = 'OTHER'

# ethnicity
main_ethn = ['WHITE', 'BLACK', 'HISPANIC', 'ASIAN']
for ethn in main_ethn:
    df.loc[df.ETHNICITY.str.startswith(ethn), 'ETHNICITY'] = ethn

df.loc[(df.ETHNICITY.notnull()) &
       (-df.ETHNICITY.isin(main_ethn)), 'ETHNICITY'] = 'OTHER'


# convert to one hot encodings
df = pd.get_dummies(df, prefix=['INSURANCE', 'LANGUAGE', 'RELIGION',
                                'MARITAL_STATUS', 'ETHNICITY', 'GENDER'])

# write back out
df.to_csv('data/final_df_numeric.csv', index=False)
