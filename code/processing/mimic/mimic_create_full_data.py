import os
from datetime import timedelta
import dask.dataframe as dd
import pandas as pd
import numpy as np


ITEM_CODES = [
    # HEART RATE
    211,  # "Heart Rate"
    220045,  # "Heart Rate"

    # Systolic/diastolic

    51,  # Arterial BP [Systolic]
    442,  # Manual BP [Systolic]
    455,  # NBP [Systolic]
    6701,  # Arterial BP #2 [Systolic]
    220179,  # Non Invasive Blood Pressure systolic
    220050,  # Arterial Blood Pressure systolic

    8368,  # Arterial BP [Diastolic]
    8440,  # Manual BP [Diastolic]
    8441,  # NBP [Diastolic]
    8555,  # Arterial BP #2 [Diastolic]
    220180,  # Non Invasive Blood Pressure diastolic
    220051,  # Arterial Blood Pressure diastolic


    # MEAN ARTERIAL PRESSURE
    456,  # "NBP Mean"
    52,  # "Arterial BP Mean"
    6702,  # Arterial BP Mean #2
    443,  # Manual BP Mean(calc)
    220052,  # "Arterial Blood Pressure mean"
    220181,  # "Non Invasive Blood Pressure mean"
    225312,  # "ART BP mean"

    # RESPIRATORY RATE
    618,  # Respiratory Rate
    615,  # Resp Rate (Total)
    220210,  # Respiratory Rate
    224690,  # Respiratory Rate (Total)


    # SPO2, peripheral
    646, 220277,

    # GLUCOSE, both lab and fingerstick
    807,  # Fingerstick Glucose
    811,  # Glucose (70-105)
    1529,  # Glucose
    3745,  # BloodGlucose
    3744,  # Blood Glucose
    225664,  # Glucose finger stick
    220621,  # Glucose (serum)
    226537,  # Glucose (whole blood)

    # TEMPERATURE
    223762,  # "Temperature Celsius"
    676,  # "Temperature C"
    223761,  # "Temperature Fahrenheit"
    678  # "Temperature F"
]


def get_stays():
    """get patients older than 16, with first stay > 48 hours"""
    admissions = dd.read_csv('data/ADMISSIONS.csv',
                             parse_dates=['ADMITTIME', 'DISCHTIME'])
    print('Inital len ADMISSIONS {}'.format(len(admissions)))
    patients = dd.read_csv('data/PATIENTS.csv', parse_dates=['DOB', 'DOD'])
    merged = dd.merge(admissions, patients, on='SUBJECT_ID')

    print('Merged with patients {}'.format(len(merged)))

    merged = merged[merged.HAS_CHARTEVENTS_DATA == 1]

    print('Only has CHARTEVENTS data {}'.format(len(merged)))

    # calculate age in years and filter to >=16 years old
    merged['AGE'] = ((merged.ADMITTIME - merged.DOB) / 365.25).dt.days
    merged_adult = merged[(merged.AGE >= 16) | (merged.AGE < 0)]

    print('Only 16+ {}'.format(len(merged_adult)))

    # filter to first stay for each patient
    two_df = merged_adult.groupby(
        'SUBJECT_ID').ADMITTIME.agg('min').reset_index()
    merged_a_first = dd.merge(merged_adult, two_df, how='inner', on=[
        'SUBJECT_ID', 'ADMITTIME'])

    print('Only first stay {}'.format(len(merged_a_first)))

    # filter to stays > 48 hour
    merged_a_f_long = merged_a_first[(merged_a_first.DISCHTIME -
                                      merged_a_first.ADMITTIME).dt.days > 2]

    print('Only longer than 48 hours {}'.format(len(merged_a_f_long)))

    # add death column
    merged_a_f_long['DIED'] = merged_a_f_long.DEATHTIME.notnull().astype('int')

    df = merged_a_f_long.drop(['ROW_ID_x', 'ROW_ID_y',
                               'DEATHTIME', 'DOD', 'DOD_HOSP',
                               'DOD_SSN', 'EXPIRE_FLAG',
                               'EDREGTIME', 'EDOUTTIME',
                               'DIAGNOSIS', 'DEATHTIME',
                               'ADMISSION_TYPE', 'ADMISSION_LOCATION',
                               'DISCHARGE_LOCATION', 'HOSPITAL_EXPIRE_FLAG',
                               'HAS_CHARTEVENTS_DATA', 'DOB'], axis=1)

    return df


def get_site(df):
    """add the site"""
    icustays = dd.read_csv('data/ICUSTAYS.csv', parse_dates=['INTIME'])

    # remove duplicate rows
    three_df = icustays.groupby(['SUBJECT_ID',
                                 'HADM_ID']).INTIME.agg('min').reset_index()
    icustays = dd.merge(icustays, three_df, how='inner', on=['SUBJECT_ID',
                                                             'HADM_ID',
                                                             'INTIME'])

    # one hot encoding of care unit
    icustays['ICU'] = icustays.FIRST_CAREUNIT

    # drop columns
    icustays = icustays.drop(['ROW_ID', 'ICUSTAY_ID', 'DBSOURCE',
                              'LAST_CAREUNIT', 'FIRST_WARDID', 'LAST_WARDID',
                              'INTIME', 'OUTTIME', 'FIRST_CAREUNIT'],
                             axis=1)

    df = dd.merge(df, icustays, how='inner', on=['SUBJECT_ID', 'HADM_ID'])
    print('After adding care unit {}'.format(len(df)))

    return df


def get_chartevents(df):
    """get the corresponding chart events"""
    if not os.path.exists('data/subset_CHARTEVENTS.csv'):
        ce = dd.read_csv('data/CHARTEVENTS.csv',
                         parse_dates=['CHARTTIME',
                                      'STORETIME'],
                         dtype={'CGID': float,
                                'ICUSTAY_ID': float,
                                'RESULTSTATUS': str,
                                'STOPPED': str,
                                'VALUE': str,
                                'VALUEUOM': str,
                                'ERROR': float,
                                'WARNING': float})
        # limit to interesting measurements
        ce_sub = ce[ce.ITEMID.isin(ITEM_CODES)]
        ce_sub = ce_sub.compute()
        ce_sub.to_csv('data/subset_CHARTEVENTS.csv', index=False)

    if not os.path.exists('data/subset_2_CHARTEVENTS.csv'):
        ce_sub = dd.read_csv('data/subset_CHARTEVENTS.csv',
                             parse_dates=['CHARTTIME',
                                          'STORETIME'],
                             dtype={'CGID': float,
                                    'ICUSTAY_ID': float,
                                    'RESULTSTATUS': str,
                                    'STOPPED': str,
                                    'VALUE': str,
                                    'VALUEUOM': str,
                                    'ERROR': float,
                                    'WARNING': float})
        hadm_ids = list(df.HADM_ID.compute())
        ce_sub_2 = ce_sub[ce_sub.HADM_ID.isin(hadm_ids)]
        ce_sub_2 = ce_sub_2.compute()
        ce_sub_2.to_csv('data/subset_2_CHARTEVENTS.csv', index=False)

    if not os.path.exists('data/subset_3_CHARTEVENTS.csv'):
        ce_sub_2 = dd.read_csv('data/subset_2_CHARTEVENTS.csv',
                               parse_dates=['CHARTTIME',
                                            'STORETIME'],
                               dtype={'CGID': float,
                                      'ICUSTAY_ID': float,
                                      'RESULTSTATUS': str,
                                      'STOPPED': str,
                                      'VALUE': str,
                                      'VALUEUOM': str,
                                      'ERROR': float,
                                      'WARNING': float})

        # remove error == 1
        ce_sub_2 = ce_sub_2[ce_sub_2.ERROR != 1]
        ce_sub_2 = ce_sub_2.drop(['Unnamed: 0', 'ROW_ID',
                                  'ICUSTAY_ID', 'STORETIME'], axis=1)

        # merge with the original data
        ce_merged = dd.merge(df, ce_sub_2, on=['SUBJECT_ID', 'HADM_ID'],
                             how='inner')

        ce_merged = ce_merged.compute()
        ce_merged.to_csv('data/subset_3_CHARTEVENTS.csv', index=False)

    if not os.path.exists('data/subset_4_CHARTEVENTS.csv'):
        ce_merged = dd.read_csv('data/subset_3_CHARTEVENTS.csv',
                                parse_dates=['CHARTTIME'],
                                dtype={'CGID': float,
                                       'RESULTSTATUS': str,
                                       'STOPPED': str,
                                       'VALUE': str,
                                       'VALUEUOM': str,
                                       'ERROR': float,
                                       'WARNING': float})

        ce_merged = ce_merged[ce_merged.WARNING != 1]
        ce_merged = ce_merged.drop(['Unnamed: 0.1', 'ERROR',
                                    'WARNING', 'RESULTSTATUS',
                                    'STOPPED'], axis=1)
        ce_merged = ce_merged.compute()
        ce_merged.to_csv('data/subset_4_CHARTEVENTS.csv', index=False)

    if not os.path.exists('data/subset_5_CHARTEVENTS.csv'):
        ce_merged = dd.read_csv('data/subset_4_CHARTEVENTS.csv',
                                parse_dates=['ADMITTIME', 'CHARTTIME'],
                                dtype={'CGID': float,
                                       'VALUE': str,
                                       'VALUEUOM': str})

        ce_merged = ce_merged.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

        ce_merged = ce_merged[ce_merged.CHARTTIME <
                              ce_merged.ADMITTIME + timedelta(days=2)]
        ce_merged = ce_merged.compute()
        ce_merged.to_csv('data/subset_5_CHARTEVENTS.csv', index=False)

    ce_final = dd.read_csv('data/subset_5_CHARTEVENTS.csv',
                           parse_dates=['ADMITTIME', 'CHARTTIME'],
                           dtype={'CGID': float,
                                  'VALUE': str,
                                  'VALUEUOM': str})

    return ce_final


def aggregate(df, item_ids, min_val, max_val, prefix):
    tmp = df[(df.ITEMID.isin(item_ids)) & (
        df.VALUENUM > min_val) & (df.VALUENUM < max_val)]

    tmp = tmp.groupby(['SUBJECT_ID', 'HADM_ID']).VALUENUM.agg(
        ['min', 'max', 'mean']).reset_index()
    tmp = tmp.rename(columns={'min': prefix + '_MIN',
                              'max': prefix + '_MAX',
                              'mean': prefix + '_MEAN'})

    return tmp


def get_vitals(df):
    """get the vitals for the first day"""

    first_day_mask = df.CHARTTIME < df.ADMITTIME + timedelta(days=1)
    second_day_mask = (df.CHARTTIME >= df.ADMITTIME + timedelta(days=1)
                       ) & (df.CHARTTIME < df.ADMITTIME + timedelta(days=2))

    df = df.drop(['ADMITTIME', 'DISCHTIME', 'INSURANCE', 'LANGUAGE',
                  'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'GENDER',
                  'AGE', 'DIED', 'CCU', 'CSRU', 'MICU', 'SICU',
                  'TSICU', 'CHARTTIME', 'CGID', 'VALUE', 'VALUEUOM'],
                 axis=1)

    # convert temp F to C
    temp_f_mask = df.ITEMID.isin([223761, 678])
    df.VALUENUM = (~temp_f_mask) * df.VALUENUM + \
        temp_f_mask * ((df.VALUENUM - 32) / 1.8)

    days = ['first', 'second']

    for d in days:
        if not os.path.exists('data/{}_day_vitals.csv'.format(d)):
            if d == 'first':
                ddf = df[first_day_mask]
            elif d == 'second':
                ddf = df[second_day_mask]
            # HR
            combined = aggregate(ddf, [211, 220045], 0, 300, 'HR')
            # sys bp
            combined = dd.merge(combined, aggregate(ddf, [51, 442, 455, 6701,
                                                          220179, 220050], 0,
                                                    400,
                                                    'SYS_BP'),
                                on=['SUBJECT_ID', 'HADM_ID'], how='outer')
            # dias bp
            combined = dd.merge(combined, aggregate(ddf, [8368, 8440, 8441,
                                                          8555, 220180,
                                                          220051], 0, 300,
                                                    'DIAS_BP'),
                                on=['SUBJECT_ID', 'HADM_ID'], how='outer')
            # mean_bp
            combined = dd.merge(combined, aggregate(ddf, [456, 52, 6702, 443,
                                                          220052, 220181,
                                                          225312],
                                                    0, 300,
                                                    'MEAN_BP'),
                                on=['SUBJECT_ID', 'HADM_ID'], how='outer')
            # resp_rate
            combined = dd.merge(combined, aggregate(ddf, [615, 618, 220210,
                                                          224690],
                                                    0, 70, 'RESP_RATE'),
                                on=['SUBJECT_ID', 'HADM_ID'], how='outer')
            # temp
            combined = dd.merge(combined, aggregate(ddf, [223761, 678,
                                                          223762, 676], 10,
                                                    50, 'TEMP'),
                                on=['SUBJECT_ID', 'HADM_ID'], how='outer')
            # spo2
            combined = dd.merge(combined, aggregate(ddf, [646, 220277], 0,
                                                    100.01, 'SPO2'),
                                on=['SUBJECT_ID', 'HADM_ID'], how='outer')
            # glucose
            combined = dd.merge(combined, aggregate(ddf, [807, 811, 1529, 3745,
                                                          3744, 225664,
                                                          220621, 226537], 0,
                                                    10000., 'GLUCOSE'),
                                on=['SUBJECT_ID', 'HADM_ID'], how='outer')

            combined = combined.compute()
            combined.to_csv('data/{}_day_vitals.csv'.format(d), index=False)


def combine_vitals():
    """combine day one and day two"""
    if not os.path.exists('data/vitals.csv'):
        first_day_vitals = dd.read_csv('data/first_day_vitals.csv').compute()
        second_day_vitals = dd.read_csv('data/second_day_vitals.csv').compute()

        vitals = dd.merge(first_day_vitals, second_day_vitals, how='outer',
                          on=['SUBJECT_ID', 'HADM_ID'],
                          suffixes=('_DAY1', '_DAY2'))

        vitals.to_csv('data/vitals.csv', index=False)

    return dd.read_csv('data/vitals.csv')


def combine_vitals_stays(df_1, df_2):
    """combine the dataframes"""
    if not os.path.exists('data/vitals_stays.csv'):
        vitals_stays = dd.merge(df_1, df_2, how='inner',
                                on=['SUBJECT_ID', 'HADM_ID'])

        vitals_stays = vitals_stays.compute()
        vitals_stays.to_csv('data/vitals_stays.csv', index=False)

    return dd.read_csv('data/vitals_stays.csv',
                       parse_dates=['ADMITTIME',
                                    'DISCHTIME'])


def add_diagnosis(vitals_stays):
    """Add diagnosis to dataset"""
    admission = dd.read_csv('data/ADMISSIONS.csv')
    diagnosis = dd.read_csv('data/DIAGNOSES_ICD.csv')

    # combine the data frames
    admission_diag = dd.merge(admission, diagnosis,
                              on=['SUBJECT_ID', 'HADM_ID'], how='outer')
    admission_diag = admission_diag.compute()

    # mask for only the patients in our data
    admission_diag = admission_diag[admission_diag.HADM_ID.isin(
        vitals_stays.HADM_ID.compute().values)]

    # convert icd9 codes
    e_mask = admission_diag.ICD9_CODE.str.startswith('E')
    # starts with 'E' and longer than 4
    admission_diag.loc[e_mask, 'ICD9_CODE'] = admission_diag.loc[
        e_mask, 'ICD9_CODE'].str[:4]

    # doesn't start with 'E' and longer than 3
    admission_diag.loc[~e_mask, 'ICD9_CODE'] = admission_diag.loc[
        ~e_mask, 'ICD9_CODE'].str[:3]

    # use crosstab to convert to binary matrix
    admission_diag = admission_diag[['HADM_ID', 'ICD9_CODE']]
    admission_diag = np.clip(pd.crosstab(admission_diag.HADM_ID,
                                         admission_diag.ICD9_CODE), 0, 1)
    admission_diag['HADM_ID'] = admission_diag.index

    final_df = dd.merge(vitals_stays, admission_diag, on='HADM_ID')

    return final_df.compute()


if __name__ == '__main__':
    # get the stays that are not excluded
    stays = get_stays()

    # get site
    sites = get_site(stays)

    ce_final = get_chartevents(sites)

    get_vitals(ce_final)
    vitals = combine_vitals()

    vitals_stays = combine_vitals_stays(sites, vitals)

    final_df = add_diagnosis(vitals_stays)

    final_df.to_csv('data/final_df.csv', index=False)
