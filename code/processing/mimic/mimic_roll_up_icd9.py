from collections import defaultdict
import pandas as pd


def read_admissions():
    """read the admissions file to get a pid-admission matching"""
    pid_adm_map = {}
    infd = open('data/mimic/ADMISSIONS.csv', 'r')
    # read the header
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admId = int(tokens[2])
        # map from admission to pid
        pid_adm_map[admId] = pid
    infd.close()

    return pid_adm_map


def icd9_ccs_mapping():
    """return map of icd9 codes to CCS values"""
    mapping = pd.read_csv('data/icd9_mapping/icd9_mapping.csv')
    return dict(zip(mapping['ICD-9-CM CODE'], mapping['CCS CATEGORY']))


def read_diagnoses(pid_adm_map, dx_rollup):
    """read in the diagnosis file and get admission diagnosis mapping"""
    pid_dx_map = defaultdict(set)
    infd = open('data/mimic/DIAGNOSES_ICD.csv', 'r')
    # read the header
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        # icd9 code
        dxStr = tokens[4][1:-1]
        if dxStr != '':
            # convert to the CCS category
            code = dx_rollup[dxStr]
            # use sets to have unique values
            pid_dx_map[pid_adm_map[admId]].add(code)
    infd.close()
    return pid_dx_map


def dict_to_bin_matrix(d):
    """convert dictionary to binary matrix"""
    df = pd.concat([
        pd.Series(list(v), name=k).astype(str) for k, v in d.items()], axis=1)
    return pd.get_dummies(df.stack()).sum(level=1).clip_upper(1)


def mimic_drop_icd9():
    """drop the previous icd9 columns"""
    mimic_df = pd.read_csv('data/mimic/mimic_raw.csv')
    # get the icd9 columns and remove them
    icd9_cols = []
    for c in mimic_df.columns:
        if c == 'ETHNICITY':
            continue
        if c[0] in ['E', 'V']:
            icd9_cols.append(c)
        try:
            int(c[0])
            icd9_cols.append(c)
        except ValueError:
            continue

    return mimic_df.drop(icd9_cols, axis=1)


if __name__ == '__main__':
    # admission to pid mapping
    pidAdmMap = read_admissions()
    # icd9 to CCS mapping
    diag_rollup = icd9_ccs_mapping()
    # using admission id to pid map and icd9 to ccs map get pid to ccs
    pidDxMap = read_diagnoses(pidAdmMap, diag_rollup)
    # convert dictionary to binary matrix
    pid_CCS_matrix = dict_to_bin_matrix(pidDxMap)
    # drop the old diagnoses
    mimic = mimic_drop_icd9()
    # add the new ones
    mimic_new = mimic.set_index('SUBJECT_ID').join(pid_CCS_matrix)
    # write the new file
    mimic_new.to_csv('data/mimic/mimic_ccs.csv', index_label='SUBJECT_ID')





