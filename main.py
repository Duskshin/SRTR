# Stanford University School of Medicine,
# NYU Grossman School of Medicine
# PI: Brian Wayda, MD MPH
# Researcher: Dusk Shin
# Project: SRTR Dataset Cleanup and Analysis
# Project: NYU Independent Project

import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def load_csv(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)

# Condenses the SRTR dataset to relevant information
def condense_csv(file_path, output_path):
    df = load_csv(file_path)

    if df is not None:
        # Takes the initial row count for calculations later
        initial_row_count = df.shape[0]
        print(f"Initial number of rows: {initial_row_count}")

        # Removes rows with blank or 'B' in 'PTR_OFFER_ACPT'
        df = df[df['PTR_OFFER_ACPT'].notna()]
        df = df[df['PTR_OFFER_ACPT'] != 'B']

        # Removes rows after a 'Y' for each unique 'DONOR_ID'
        df_condensed = pd.DataFrame()

        donor_groups = df.groupby('DONOR_ID')
        total_donors = len(donor_groups)

        for donor_id, group in tqdm(donor_groups, total=total_donors, desc="Processing donors"):
            rows_to_keep = []
            found_y = False

            for index, row in group.iterrows():
                rows_to_keep.append(row)
                if row['PTR_OFFER_ACPT'] == 'Y':
                    found_y = True
                    break

            df_condensed = pd.concat([df_condensed, pd.DataFrame(rows_to_keep)], ignore_index=True)

        final_row_count = df_condensed.shape[0]
        print(f"Number of rows after condensing: {final_row_count}")

        unique_donors = df_condensed['DONOR_ID'].nunique()
        print(f"Number of unique donors: {unique_donors}")

        # Counting number of Ys, which are acceptances
        y_count = df_condensed['PTR_OFFER_ACPT'].value_counts().get('Y', 0)
        print(f"Number of 'Y's: {y_count}")

        df_condensed.to_csv(output_path, index=False)
        print(f"Condensed CSV exported as {output_path}")

# Cleans up dataset to relevant information
def clean_up_offer_dataset(file_path_1, file_path_2, file_path_3, output_path):
    df_1 = load_csv(file_path_1)
    df_2 = load_csv(file_path_2)
    df_3 = load_csv(file_path_3)

    if df_1 is not None and df_2 is not None and df_3 is not None:
        # MErging df_1 with df_2 to get recipient information
        df_1 = df_1.merge(df_2[['PX_ID', 'age_listing', 'female', 'height_cm', 'weight_kg']], on='PX_ID', how='left')

        # Merge df_1 with df_3 to get donor information
        df_1 = df_1.merge(df_3[['DONOR_ID', 'DON_AGE', 'don_female', 'DON_HGT_CM', 'DON_WGT_KG']], on='DONOR_ID', how='left')

        # This calculates PHM for donors and recipients
        def calculate_phm(age, height_cm, weight_kg, female, is_donor=True):
            height_m = height_cm / 100  # convert height to meters
            if is_donor:
                lvm_a = 6.82 if female == 1 else 8.25
                rvm_a = 10.59 if female == 1 else 11.25
            else:
                lvm_a = 6.82 if female == 1 else 8.25
                rvm_a = 10.59 if female == 1 else 11.25
            lvm = lvm_a * (height_m ** 0.54) * (weight_kg ** 0.61)
            if age == 0:
                rvm = 0  # Skip RVM calculation if age is 0
            else:
                rvm = rvm_a * (age ** -0.32) * (height_m ** 1.135) * (weight_kg ** 0.315)
            return lvm + rvm

        df_1['donor_PHM'] = df_1.apply(lambda x: calculate_phm(x['DON_AGE'], x['DON_HGT_CM'], x['DON_WGT_KG'], x['don_female'], is_donor=True), axis=1)
        df_1['recipient_PHM'] = df_1.apply(lambda x: calculate_phm(x['age_listing'], x['height_cm'], x['weight_kg'], x['female'], is_donor=False), axis=1)

        df_1['PHM_ratio'] = df_1['donor_PHM'] / df_1['recipient_PHM'] # Calculates PHM ratio

        df_1 = df_1[df_1['PHM_ratio'].notna()] # Removes rows where PHM_ratio is blank

        # New columns based on conditions for viable offer
        df_1['size_mismatched'] = df_1['PHM_ratio'].apply(lambda x: 1 if x < 0.8 or x > 1.3 else 0)
        df_1['recipient_unavailable'] = df_1['PTR_PRIME_OPO_REFUSAL_ID'].apply(lambda x: 1 if (720 <= x < 730) or (800 <= x <= 809) else 0)
        df_1['incompatible'] = df_1['PTR_PRIME_OPO_REFUSAL_ID'].apply(lambda x: 1 if (730 <= x < 740) or (810 <= x <= 819) else 0)
        df_1['too_far'] = df_1['PTR_DISTANCE'].apply(lambda x: 1 if x >= 1000 else 0)

        # Add 'viable_offer' column tothe new CSV
        df_1['viable_offer'] = df_1.apply(lambda x: 1 if x['size_mismatched'] == 0 and x['recipient_unavailable'] == 0 and x['incompatible'] == 0 and x['too_far'] == 0 else 0, axis=1)

        # Create 'used_ht' column
        donor_ids_in_file_2 = set(df_2['DONOR_ID'].unique())
        df_1['used_ht'] = df_1['DONOR_ID'].apply(lambda x: 1 if x in donor_ids_in_file_2 else 0)

        total_offers = df_1.shape[0]
        viable_offers = df_1['viable_offer'].sum()
        viable_percentage = (viable_offers / total_offers) * 100
        print(f"Percentage of viable offers: {viable_percentage:.2f}%")

        df_1.to_csv(output_path, index=False)
        print(f"Created CSV with PHM_ratio, viable_offer, and used_ht calculations exported as {output_path}")

# This creates the analytical cohort which is then used for the study
def create_analytical_cohort(file_path, output_path):
    df = load_csv(file_path)

    if df is not None:
        df = df[df['age_listing'] >= 18]
        df = df[(df['delisted'] != 0) | (df['died_wl'] != 0)]
        df['rem_dt'] = pd.to_datetime(df['rem_dt'], errors='coerce')
        df = df[df['rem_dt'].dt.year == 2022]
        df.to_csv(output_path, index=False)
        print(f"Created analytical cohort CSV exported as {output_path}")

# Condenses the offerset even further
def condense_cleaned_offer_dataset(file_path_1, file_path_2, output_path):
    df_1 = load_csv(file_path_1)
    df_2 = load_csv(file_path_2)

    if df_1 is not None and df_2 is not None:
        df_merged = df_1.merge(df_2[['PX_ID', 'rem_dt']], on='PX_ID', how='inner')
        df_merged['rem_dt'] = pd.to_datetime(df_merged['rem_dt'], errors='coerce')
        df_merged['MATCH_SUBMIT_DT'] = pd.to_datetime(df_merged['MATCH_SUBMIT_DT'], format='%d%b%Y:%H:%M:%S.%f',
                                                      errors='coerce')
        df_merged = df_merged[(df_merged['rem_dt'] - df_merged['MATCH_SUBMIT_DT']).dt.days <= 365]
        df_merged.to_csv(output_path, index=False)
        print(f"Created condensed cleaned offer dataset CSV exported as {output_path}")

# Updating the analytical cohort with additional conditionals
def update_analytical_cohort(file_path_1, file_path_2, output_path):
    df_1 = load_csv(file_path_1)  # analyticalcohort.csv
    df_2 = load_csv(file_path_2)  # match2021and2022condensed.csv

    if df_1 is not None and df_2 is not None:
        df_1['num_offers_total'] = 0
        df_1['num_offers_viable'] = 0
        df_1['num_size_mismatched'] = 0
        df_1['num_recipient_unavail'] = 0
        df_1['num_incompatible'] = 0
        df_1['num_too_far'] = 0
        df_1['num_used_ht'] = 0
        df_1['num_viable_and_used_ht'] = 0

        offer_counts = df_2.groupby('PX_ID').agg(
            num_offers_total=pd.NamedAgg(column='PX_ID', aggfunc='count'),
            num_offers_viable=pd.NamedAgg(column='viable_offer', aggfunc='sum'),
            num_size_mismatched=pd.NamedAgg(column='size_mismatched', aggfunc='sum'),
            num_recipient_unavail=pd.NamedAgg(column='recipient_unavailable', aggfunc='sum'),
            num_incompatible=pd.NamedAgg(column='incompatible', aggfunc='sum'),
            num_too_far=pd.NamedAgg(column='too_far', aggfunc='sum'),
            num_used_ht=pd.NamedAgg(column='used_ht', aggfunc='sum'),
            num_viable_and_used_ht=pd.NamedAgg(column='viable_offer', aggfunc=lambda x: ((df_2.loc[x.index, 'viable_offer'] == 1) & (df_2.loc[x.index, 'used_ht'] == 1)).sum())
        ).reset_index()

        df_1 = df_1.merge(offer_counts, on='PX_ID', how='left')
        df_1.fillna(0, inplace=True)
        df_1 = df_1[[col for col in df_1.columns if not col.endswith('_x')]] # removes 'x'
        df_1.columns = [col.replace('_y', '') for col in df_1.columns]
        df_1.to_csv(output_path, index=False)
        print(f"Updated analytical cohort CSV exported as {output_path}")

# Analyzing viable offers which is defined by those that are
# delisted or died on the waitlist
def analyze_viable_offers(file_path):
    df = load_csv(file_path)

    if df is not None:
        viable_and_died_or_delisted = \
        df[(df['num_offers_viable'] >= 1) & ((df['died_wl'] == 1) | (df['delisted'] == 1))].shape[0]
        print(
            f"Number of PX_IDs with num_offers_viable >= 1 and died_wl = 1 or delisted = 1: {viable_and_died_or_delisted}")
# Calculate the mean and standard deviation for a given column.
def calculate_mean_sd(df, column_name):
    mean = df[column_name].mean()
    sd = df[column_name].std()
    return f"{mean:.2f} / {sd:.2f}"

# Calculate the percentage of rows where the column equals the condition_value
def calculate_percentage(df, column_name, condition_value=1):
    return (df[column_name] == condition_value).mean() * 100 # turning it into a percentage

# Calculate the median, minimum, and maximum for a given column
def calculate_median_range(df, column_name):
    median = df[column_name].median()
    min_value = df[column_name].min()
    max_value = df[column_name].max()
    return f"{median:.2f} ({min_value:.2f}, {max_value:.2f})"

# Creating a table for analytical variables used for first abstract
def create_table_1(file_path_1, file_path_2, file_path_3, file_path_4, output_path):
    df_1 = load_csv(file_path_1)  # updated_analyticalcohort.csv
    df_2 = load_csv(file_path_2)  # match2021and2022condensed.csv
    df_ctr = load_csv(file_path_3)  # ctr.csv
    df_inst = load_csv(file_path_4)  # institutions.csv

    if df_1 is not None and df_2 is not None and df_ctr is not None and df_inst is not None:
        df_1['rem_dt'] = pd.to_datetime(df_1['rem_dt'], errors='coerce')
        df_1['list_dt'] = pd.to_datetime(df_1['list_dt'], errors='coerce')
        df_1['time_on_waitlist'] = (df_1['rem_dt'] - df_1['list_dt']).dt.days
        df_a = df_1[df_1['num_viable_and_used_ht'] > 1]
        df_b = df_1[df_1['num_viable_and_used_ht'] == 0]
        df_a = df_a.merge(df_2[['PX_ID', 'recipient_PHM']], on='PX_ID', how='left')
        df_b = df_b.merge(df_2[['PX_ID', 'recipient_PHM']], on='PX_ID', how='left')
        df_a = df_a.merge(df_ctr[['ctr_listing', 'Volume']], on='ctr_listing', how='left')
        df_b = df_b.merge(df_ctr[['ctr_listing', 'Volume']], on='ctr_listing', how='left')
        df_a = df_a.merge(df_inst[['CTR_ID', 'REGION']], left_on='ctr_listing', right_on='CTR_ID', how='left')
        df_b = df_b.merge(df_inst[['CTR_ID', 'REGION']], left_on='ctr_listing', right_on='CTR_ID', how='left')

        # Define Candidate Characteristics
        candidate_char = [
            "Age (mean/SD)",
            "Female (%)",
            "White (%)",
            "Black (%)",
            "Asian (%)",
            "Hispanic (%)",
            "Other (%)",
            "Type A (%)",
            "Type B (%)",
            "Type AB (%)",
            "Type O (%)",
            "Height (mean/SD)",
            "Weight (mean/SD)",
            "PHM (mean/SD)",
            "BMI (mean/SD)",
            "NIDCM (%)",
            "Ischemic (%)",
            "RCM (%)",
            "HCM (%)",
            "Congenital (%)",
            "Valvular (%)",
            "Retx_gf (%)",
            "Other Etiology (%)",
            "Diabetes (%)",
            "Renal Moderate Listing (%)",
            "Renal Severe Listing (%)",
            "Renal Mild Listing (%)",
            "Renal Normal Listing (%)",
            "Prior CSurg (%)",
            "InitStat Old1A (%)",
            "InitStat Old1B (%)",
            "InitStat Old1 (%)",
            "InitStat Old2 (%)",
            "InitStat New1 (%)",
            "InitStat New2 (%)",
            "InitStat New3 (%)",
            "InitStat New4 (%)",
            "InitStat New5 (%)",
            "InitStat New6 (%)",
            "LastStat Old1A (%)",
            "LastStat Old1B (%)",
            "LastStat Old1 (%)",
            "LastStat Old2 (%)",
            "LastStat New1 (%)",
            "LastStat New2 (%)",
            "LastStat New3 (%)",
            "LastStat New4 (%)",
            "LastStat New5 (%)",
            "LastStat New6 (%)",
            "LastStat Inactive (%)",
            "Volume - Low (%)",
            "Volume - Medium (%)",
            "Volume - High (%)",
            "Region 1 (%)",
            "Region 2 (%)",
            "Region 3 (%)",
            "Region 4 (%)",
            "Region 5 (%)",
            "Region 6 (%)",
            "Region 7 (%)",
            "Region 8 (%)",
            "Region 9 (%)",
            "Region 10 (%)",
            "Region 11 (%)",
            "Time on Waitlist (median, range)",
            "num_offers_total (median, range)",
            "num_offers_viable (median, range)",
            "num_size_mismatched (median, range)",
            "num_recipient_unavail (median, range)",
            "num_incompatible (median, range)",
            "num_too_far (median, range)",
            "num_used_ht (median, range)",
            "num_viable_and_used_ht (median, range)"
        ]

        # Calculations for Column A
        column_a = [
            calculate_mean_sd(df_a, 'age_listing'),
            calculate_percentage(df_a, 'female'),
            calculate_percentage(df_a, 'white'),
            calculate_percentage(df_a, 'black'),
            calculate_percentage(df_a, 'asian'),
            calculate_percentage(df_a, 'hispanic_latino'),
            calculate_percentage(df_a, 'other_race'),
            calculate_percentage(df_a, 'typeA'),
            calculate_percentage(df_a, 'typeB'),
            calculate_percentage(df_a, 'typeAB'),
            calculate_percentage(df_a, 'typeO'),
            calculate_mean_sd(df_a, 'height_cm'),
            calculate_mean_sd(df_a, 'weight_kg'),
            calculate_mean_sd(df_a, 'recipient_PHM'),
            calculate_mean_sd(df_a, 'bmi'),
            calculate_percentage(df_a, 'nidcm'),
            calculate_percentage(df_a, 'ischemic'),
            calculate_percentage(df_a, 'rcm'),
            calculate_percentage(df_a, 'hcm'),
            calculate_percentage(df_a, 'congenital'),
            calculate_percentage(df_a, 'valvular'),
            calculate_percentage(df_a, 'retx_gf'),
            calculate_percentage(df_a, 'other_etiology'),
            calculate_percentage(df_a, 'diabetes'),
            calculate_percentage(df_a, 'renal_moderate_listing'),
            calculate_percentage(df_a, 'renal_severe_listing'),
            calculate_percentage(df_a, 'renal_mild_listing'),
            calculate_percentage(df_a, 'renal_normal_listing'),
            calculate_percentage(df_a, 'prior_csurg'),
            calculate_percentage(df_a, 'initstat_old1a'),
            calculate_percentage(df_a, 'initstat_old1b'),
            calculate_percentage(df_a, 'initstat_old1'),
            calculate_percentage(df_a, 'initstat_old2'),
            calculate_percentage(df_a, 'initstat_new1'),
            calculate_percentage(df_a, 'initstat_new2'),
            calculate_percentage(df_a, 'initstat_new3'),
            calculate_percentage(df_a, 'initstat_new4'),
            calculate_percentage(df_a, 'initstat_new5'),
            calculate_percentage(df_a, 'initstat_new6'),
            calculate_percentage(df_a, 'laststat_old1a'),
            calculate_percentage(df_a, 'laststat_old1b'),
            calculate_percentage(df_a, 'laststat_old1'),
            calculate_percentage(df_a, 'laststat_old2'),
            calculate_percentage(df_a, 'laststat_new1'),
            calculate_percentage(df_a, 'laststat_new2'),
            calculate_percentage(df_a, 'laststat_new3'),
            calculate_percentage(df_a, 'laststat_new4'),
            calculate_percentage(df_a, 'laststat_new5'),
            calculate_percentage(df_a, 'laststat_new6'),
            calculate_percentage(df_a, 'laststat_inactive'),
            calculate_percentage(df_a, 'Volume', 'low'),
            calculate_percentage(df_a, 'Volume', 'medium'),
            calculate_percentage(df_a, 'Volume', 'large'),
            calculate_percentage(df_a, 'REGION', 1),
            calculate_percentage(df_a, 'REGION', 2),
            calculate_percentage(df_a, 'REGION', 3),
            calculate_percentage(df_a, 'REGION', 4),
            calculate_percentage(df_a, 'REGION', 5),
            calculate_percentage(df_a, 'REGION', 6),
            calculate_percentage(df_a, 'REGION', 7),
            calculate_percentage(df_a, 'REGION', 8),
            calculate_percentage(df_a, 'REGION', 9),
            calculate_percentage(df_a, 'REGION', 10),
            calculate_percentage(df_a, 'REGION', 11),
            calculate_median_range(df_a, 'time_on_waitlist'),
            calculate_median_range(df_a, 'num_offers_total'),
            calculate_median_range(df_a, 'num_offers_viable'),
            calculate_median_range(df_a, 'num_size_mismatched'),
            calculate_median_range(df_a, 'num_recipient_unavail'),
            calculate_median_range(df_a, 'num_incompatible'),
            calculate_median_range(df_a, 'num_too_far'),
            calculate_median_range(df_a, 'num_used_ht'),
            calculate_median_range(df_a, 'num_viable_and_used_ht')
        ]

        # Calculations for Column B
        column_b = [
            calculate_mean_sd(df_b, 'age_listing'),
            calculate_percentage(df_b, 'female'),
            calculate_percentage(df_b, 'white'),
            calculate_percentage(df_b, 'black'),
            calculate_percentage(df_b, 'asian'),
            calculate_percentage(df_b, 'hispanic_latino'),
            calculate_percentage(df_b, 'other_race'),
            calculate_percentage(df_b, 'typeA'),
            calculate_percentage(df_b, 'typeB'),
            calculate_percentage(df_b, 'typeAB'),
            calculate_percentage(df_b, 'typeO'),
            calculate_mean_sd(df_b, 'height_cm'),
            calculate_mean_sd(df_b, 'weight_kg'),
            calculate_mean_sd(df_b, 'recipient_PHM'),
            calculate_mean_sd(df_b, 'bmi'),
            calculate_percentage(df_b, 'nidcm'),
            calculate_percentage(df_b, 'ischemic'),
            calculate_percentage(df_b, 'rcm'),
            calculate_percentage(df_b, 'hcm'),
            calculate_percentage(df_b, 'congenital'),
            calculate_percentage(df_b, 'valvular'),
            calculate_percentage(df_b, 'retx_gf'),
            calculate_percentage(df_b, 'other_etiology'),
            calculate_percentage(df_b, 'diabetes'),
            calculate_percentage(df_b, 'renal_moderate_listing'),
            calculate_percentage(df_b, 'renal_severe_listing'),
            calculate_percentage(df_b, 'renal_mild_listing'),
            calculate_percentage(df_b, 'renal_normal_listing'),
            calculate_percentage(df_b, 'prior_csurg'),
            calculate_percentage(df_b, 'initstat_old1a'),
            calculate_percentage(df_b, 'initstat_old1b'),
            calculate_percentage(df_b, 'initstat_old1'),
            calculate_percentage(df_b, 'initstat_old2'),
            calculate_percentage(df_b, 'initstat_new1'),
            calculate_percentage(df_b, 'initstat_new2'),
            calculate_percentage(df_b, 'initstat_new3'),
            calculate_percentage(df_b, 'initstat_new4'),
            calculate_percentage(df_b, 'initstat_new5'),
            calculate_percentage(df_b, 'initstat_new6'),
            calculate_percentage(df_b, 'laststat_old1a'),
            calculate_percentage(df_b, 'laststat_old1b'),
            calculate_percentage(df_b, 'laststat_old1'),
            calculate_percentage(df_b, 'laststat_old2'),
            calculate_percentage(df_b, 'laststat_new1'),
            calculate_percentage(df_b, 'laststat_new2'),
            calculate_percentage(df_b, 'laststat_new3'),
            calculate_percentage(df_b, 'laststat_new4'),
            calculate_percentage(df_b, 'laststat_new5'),
            calculate_percentage(df_b, 'laststat_new6'),
            calculate_percentage(df_b, 'laststat_inactive'),
            calculate_percentage(df_b, 'Volume', 'low'),
            calculate_percentage(df_b, 'Volume', 'medium'),
            calculate_percentage(df_b, 'Volume', 'large'),
            calculate_percentage(df_b, 'REGION', 1),
            calculate_percentage(df_b, 'REGION', 2),
            calculate_percentage(df_b, 'REGION', 3),
            calculate_percentage(df_b, 'REGION', 4),
            calculate_percentage(df_b, 'REGION', 5),
            calculate_percentage(df_b, 'REGION', 6),
            calculate_percentage(df_b, 'REGION', 7),
            calculate_percentage(df_b, 'REGION', 8),
            calculate_percentage(df_b, 'REGION', 9),
            calculate_percentage(df_b, 'REGION', 10),
            calculate_percentage(df_b, 'REGION', 11),
            calculate_median_range(df_b, 'time_on_waitlist'),
            calculate_median_range(df_b, 'num_offers_total'),
            calculate_median_range(df_b, 'num_offers_viable'),
            calculate_median_range(df_b, 'num_size_mismatched'),
            calculate_median_range(df_b, 'num_recipient_unavail'),
            calculate_median_range(df_b, 'num_incompatible'),
            calculate_median_range(df_b, 'num_too_far'),
            calculate_median_range(df_b, 'num_used_ht'),
            calculate_median_range(df_b, 'num_viable_and_used_ht')
        ]

        table_1 = pd.DataFrame({
            "Candidate Char.": candidate_char,
            "Column A": column_a,
            "Column B": column_b
        })

        table_1.to_csv(output_path, index=False)
        print(f"Table 1 exported as {output_path}")


def plot_pie_chart_seaborn(labels, sizes, title, subplot_title, ax):
    colors = sns.color_palette('pastel')[0:len(labels)]
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title(subplot_title)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.setp(autotexts, size=10, weight="bold", color="black")


def create_graphs(file_path_1):
    df_table = pd.read_csv(file_path_1)

    df_table['Column A'] = pd.to_numeric(df_table['Column A'], errors='coerce')
    df_table['Column B'] = pd.to_numeric(df_table['Column B'], errors='coerce')

    sns.set(style="whitegrid")

    fig, axs = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle('Demographic of Those Delisted Due to Deterioration or Died on the Waitlist', fontsize=16)

    labels = ['Female', 'Male']
    sizes_a = [df_table.loc[df_table['Candidate Char.'] == 'Female (%)', 'Column A'].values[0],
               100 - df_table.loc[df_table['Candidate Char.'] == 'Female (%)', 'Column A'].values[0]]
    sizes_b = [df_table.loc[df_table['Candidate Char.'] == 'Female (%)', 'Column B'].values[0],
               100 - df_table.loc[df_table['Candidate Char.'] == 'Female (%)', 'Column B'].values[0]]

    plot_pie_chart_seaborn(labels, sizes_a, "Gender Distribution", "Column A: Receiving at Least One Viable Offer",
                           axs[0, 0])
    plot_pie_chart_seaborn(labels, sizes_b, "Gender Distribution", "Column B: Receiving No Viable Offer", axs[0, 1])

    race_labels = ['White', 'Black', 'Asian', 'Hispanic', 'Other']
    sizes_a = [df_table.loc[df_table['Candidate Char.'] == race + " (%)", 'Column A'].values[0] for race in race_labels]
    sizes_b = [df_table.loc[df_table['Candidate Char.'] == race + " (%)", 'Column B'].values[0] for race in race_labels]

    plot_pie_chart_seaborn(race_labels, sizes_a, "Race Distribution", "Column A: Receiving at Least One Viable Offer",
                           axs[1, 0])
    plot_pie_chart_seaborn(race_labels, sizes_b, "Race Distribution", "Column B: Receiving No Viable Offer", axs[1, 1])

    blood_labels = ['Type A', 'Type B', 'Type AB', 'Type O']
    sizes_a = [df_table.loc[df_table['Candidate Char.'] == blood_type + " (%)", 'Column A'].values[0] for blood_type in
               blood_labels]
    sizes_b = [df_table.loc[df_table['Candidate Char.'] == blood_type + " (%)", 'Column B'].values[0] for blood_type in
               blood_labels]

    plot_pie_chart_seaborn(blood_labels, sizes_a, "Blood Type Distribution",
                           "Column A: Receiving at Least One Viable Offer", axs[2, 0])
    plot_pie_chart_seaborn(blood_labels, sizes_b, "Blood Type Distribution", "Column B: Receiving No Viable Offer",
                           axs[2, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
    plt.savefig('demographic_pie_charts.png')
    plt.show()


def create_bar_charts(file_path_2):
    df = pd.read_csv(file_path_2)

    sns.set(style="whitegrid")

    df_a = df[df['num_viable_and_used_ht'] >= 1] # Column A,
    df_b = df[df['num_viable_and_used_ht'] == 0] # Column B, 

    fig, axs = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle('Demographic Distribution in Updated Analytical Cohort', fontsize=16)

    gender_counts_a = df_a['female'].value_counts(normalize=True) * 100
    sns.barplot(x=gender_counts_a.index, y=gender_counts_a.values, ax=axs[0, 0], palette="pastel")
    axs[0, 0].set_title('Gender Distribution - Column A: Receiving at Least One Viable Offer')
    axs[0, 0].set_xticklabels(['Male', 'Female'])
    axs[0, 0].set_ylabel('Percentage (%)')

    gender_counts_b = df_b['female'].value_counts(normalize=True) * 100
    sns.barplot(x=gender_counts_b.index, y=gender_counts_b.values, ax=axs[0, 1], palette="pastel")
    axs[0, 1].set_title('Gender Distribution - Column B: Receiving No Viable Offer')
    axs[0, 1].set_xticklabels(['Male', 'Female'])
    axs[0, 1].set_ylabel('Percentage (%)')

    race_labels = ['white', 'black', 'asian', 'hispanic_latino', 'other_race']
    race_counts_a = df_a[race_labels].sum() / len(df_a) * 100
    sns.barplot(x=race_labels, y=race_counts_a.values, ax=axs[1, 0], palette="pastel")
    axs[1, 0].set_title('Race Distribution - Column A: Receiving at Least One Viable Offer')
    axs[1, 0].set_ylabel('Percentage (%)')
    axs[1, 0].set_xticklabels(['White', 'Black', 'Asian', 'Hispanic', 'Other'], rotation=45)

    race_counts_b = df_b[race_labels].sum() / len(df_b) * 100
    sns.barplot(x=race_labels, y=race_counts_b.values, ax=axs[1, 1], palette="pastel")
    axs[1, 1].set_title('Race Distribution - Column B: Receiving No Viable Offer')
    axs[1, 1].set_ylabel('Percentage (%)')
    axs[1, 1].set_xticklabels(['White', 'Black', 'Asian', 'Hispanic', 'Other'], rotation=45)

    blood_labels = ['typeA', 'typeB', 'typeAB', 'typeO']
    blood_counts_a = df_a[blood_labels].sum() / len(df_a) * 100
    sns.barplot(x=blood_labels, y=blood_counts_a.values, ax=axs[2, 0], palette="pastel")
    axs[2, 0].set_title('Blood Type Distribution - Column A: Receiving at Least One Viable Offer')
    axs[2, 0].set_ylabel('Percentage (%)')
    axs[2, 0].set_xticklabels(['Type A', 'Type B', 'Type AB', 'Type O'])

    blood_counts_b = df_b[blood_labels].sum() / len(df_b) * 100
    sns.barplot(x=blood_labels, y=blood_counts_b.values, ax=axs[2, 1], palette="pastel")
    axs[2, 1].set_title('Blood Type Distribution - Column B: Receiving No Viable Offer')
    axs[2, 1].set_ylabel('Percentage (%)')
    axs[2, 1].set_xticklabels(['Type A', 'Type B', 'Type AB', 'Type O'])

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
    plt.savefig('demographic_bar_charts_A_B.png')
    plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression calculations from tbale 1
def logistic_regression(X, y, num_steps, learning_rate, add_intercept=True):
    if add_intercept:
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)

    weights = np.zeros(X.shape[1])

    for step in range(num_steps):
        z = np.dot(X, weights)
        predictions = sigmoid(z)
        gradient = np.dot(X.T, predictions - y) / y.size
        weights -= learning_rate * gradient

        if step % 10000 == 0:
            print(log_likelihood(X, y, weights))

    return weights

def log_likelihood(X, y, weights):
    z = np.dot(X, weights)
    ll = np.sum(y*z - np.log(1 + np.exp(z)))
    return ll

def plot_coefficients(weights, feature_names, output_path):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, weights)
    plt.xlabel("Coefficient")
    plt.title("Logistic Regression Coefficients")
    plt.savefig(f"{output_path}_logistic_regression_coefficients.png")
    plt.show()

def plot_roc_curve(y_true, y_pred_prob, output_path):
    thresholds = np.arange(0.0, 1.1, 0.01)
    tpr = []
    fpr = []

    for threshold in thresholds:
        y_pred = y_pred_prob >= threshold
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='orange', lw=2)
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.savefig(f"{output_path}_logistic_regression_roc_curve.png")
    plt.show()

# test function 1
def create_function_10(file_path_1, file_path_2, output_prefix):
    df_table = pd.read_csv(file_path_1)
    df_analytical = pd.read_csv(file_path_2)

    predictors = ['age_listing', 'female', 'white', 'black', 'asian',
                  'hispanic_latino', 'other_race', 'height_cm', 'weight_kg']

    X = df_analytical[predictors].values
    y = df_analytical['num_viable_and_used_ht'].apply(lambda x: 1 if x >= 1 else 0).values

    weights = logistic_regression(X, y, num_steps=300000, learning_rate=0.01)

    plot_coefficients(weights, ['Intercept'] + predictors, output_prefix)

    y_pred_prob = sigmoid(np.dot(np.insert(X, 0, 1, axis=1), weights))
    plot_roc_curve(y, y_pred_prob, output_prefix)

    coefficients = pd.DataFrame({'Predictor': ['Intercept'] + predictors, 'Coefficient': weights})
    coefficients.to_csv(f"{output_prefix}_logistic_regression_coefficients.csv", index=False)
    print(f"Logistic regression coefficients saved to {output_prefix}_logistic_regression_coefficients.csv")

# test function 2
def create_function_11(file_path, output_path):
    df = pd.read_csv(file_path)

    df['Age (<40)'] = (df['age_listing'] < 40).astype(int)
    df['Age (40-59)'] = ((df['age_listing'] >= 40) & (df['age_listing'] <= 59)).astype(int)
    df['Age (60+)'] = (df['age_listing'] >= 60).astype(int)

    df['rcm_hcm'] = ((df['rcm'] == 1) | (df['hcm'] == 1)).astype(int)
    df['other_combines'] = ((df['valvular'] == 1) | (df['retx_gf'] == 1) | (df['other_etiology'] == 1)).astype(int)
    df['renal_mod_sev'] = ((df['renal_moderate_listing'] == 1) | (df['renal_severe_listing'] == 1)).astype(int)

    df['BMI (<20)'] = (df['weight_kg'] / ((df['height_cm'] / 100) ** 2) < 20).astype(int)
    df['BMI (>=30)'] = (df['weight_kg'] / ((df['height_cm'] / 100) ** 2) >= 30).astype(int)

    # Mapping variable names to DataFrame columns
    variables = {
        'Age (<40)': 'Age (<40)',
        'Age (40-59)': 'Age (40-59)',
        'Age (60+)': 'Age (60+)',
        'Female (%)': 'female',
        'White (%)': 'white',
        'Black (%)': 'black',
        'Asian (%)': 'asian',
        'Hispanic (%)': 'hispanic_latino',
        'Other (%)': 'other_race',
        'Type A (%)': 'typeA',
        'Type B (%)': 'typeB',
        'Type AB (%)': 'typeAB',
        'Type O (%)': 'typeO',
        'BMI (<20)': 'BMI (<20)',
        'BMI (>=30)': 'BMI (>=30)',
        'NIDCM (%)': 'nidcm',
        'Ischemic (%)': 'ischemic',
        'rcm_hcm (%)': 'rcm_hcm',
        'Congenital (%)': 'congenital',
        'other_combines (%)': 'other_combines',
        'Diabetes (%)': 'diabetes',
        'renal_mod_sev (%)': 'renal_mod_sev',
        'Renal Mild Listing (%)': 'renal_mild_listing',
        'Renal Normal Listing (%)': 'renal_normal_listing',
        'Prior CSurg (%)': 'prior_csurg',
        'InitStat Old1A (%)': 'initstat_old1a',
        'InitStat Old1B (%)': 'initstat_old1b',
        'InitStat Old2 (%)': 'initstat_old2',
        'InitStat New 1 (%)': 'initstat_new1',
        'InitStat New 2 (%)': 'initstat_new2',
        'InitStat New 3 (%)': 'initstat_new3',
        'InitStat New 4 (%)': 'initstat_new4',
        'InitStat New 5 (%)': 'initstat_new5',
        'InitStat New 6 (%)': 'initstat_new6',
        'Volume - Low (%)': 'volume_low',
        'Volume - Medium (%)': 'volume_medium',
        'Volume - High (%)': 'volume_high'
    }

    results_df = pd.DataFrame(columns=['Candidate Char.', '% of the total analytic sample (n=)',
                                       'Received viable offer (n=; x%)', 'Did not receive viable offer (n=; x%)',
                                       'p-value'])

    total_sample = len(df)

    for name, col in variables.items():
        viable = df[df['num_viable_and_used_ht'] >= 1]
        non_viable = df[df['num_viable_and_used_ht'] == 0]

        # Calculate counts based on specific columns
        if col in df.columns:  # Column exists in the DataFrame
            n_viable = viable[col].sum()
            n_non_viable = non_viable[col].sum()
        else:  # If column not found, skip
            continue

        viable_percent = round((n_viable / (n_viable + n_non_viable) * 100), 2) if (n_viable + n_non_viable) != 0 else 0
        non_viable_percent = 100 - viable_percent

        # Construct the correct observed contingency table
        observed = np.array([[n_viable, len(viable) - n_viable],
                             [n_non_viable, len(non_viable) - n_non_viable]])

        # Calculate chi-squared and p-value
        chi2, p_value = calculate_chi_squared(observed)

        new_row = pd.DataFrame({
            'Candidate Char.': [name],
            '% of the total analytic sample (n=)': f"{df[col].sum()} ({round(df[col].sum() / total_sample * 100, 2)}%)",
            'Received viable offer (n=; x%)': f"{n_viable} ({viable_percent:.2f}%)",
            'Did not receive viable offer (n=; x%)': f"{n_non_viable} ({non_viable_percent:.2f}%)",
            'p-value': f"{p_value:.2e}" if p_value < 0.0001 else round(p_value, 4)
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    total_viable = len(df[df['num_viable_and_used_ht'] >= 1])
    total_non_viable = len(df[df['num_viable_and_used_ht'] == 0])
    total_viable_percent = round((total_viable / total_sample) * 100, 2)
    total_non_viable_percent = 100 - total_viable_percent

    totals_row = pd.DataFrame({
        'Candidate Char.': ['Totals'],
        '% of the total analytic sample (n=)': f"{total_sample} (100.00%)",
        'Received viable offer (n=; x%)': f"{total_viable} ({total_viable_percent:.2f}%)",
        'Did not receive viable offer (n=; x%)': f"{total_non_viable} ({total_non_viable_percent:.2f}%)",
        'p-value': 'N/A'  # p-value is not applicable for the totals row
    })

    results_df = pd.concat([totals_row, results_df], ignore_index=True)
    results_df.to_csv(output_path, index=False)
    print(f"Analysis table saved to {output_path}")


def calculate_chi_squared(observed):
    """Calculate chi-squared statistic and p-value."""
    total = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    # Avoid division by zero in expected
    expected = np.where(expected == 0, 1e-10, expected)
    chi_squared = ((observed - expected) ** 2 / expected).sum()
    # Calculate degrees of freedom and p-value
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    p_value = chi2_cdf(chi_squared, dof)
    return chi_squared, p_value


# Chi-squared cumulative distribution function
def chi2_cdf(x, df):
    """Calculate chi-squared CDF using a series expansion."""
    k = df / 2.0
    return (np.exp(-x / 2) * sum((x / 2) ** i / math.factorial(i) for i in range(int(k) + 1)))

# refusal codes testing
def create_function_12(refusal_file, analytical_file, match_file, output_file):
    df_refusal = pd.read_csv(refusal_file, encoding='ISO-8859-1')
    df_analytical = pd.read_csv(analytical_file, encoding='ISO-8859-1')
    df_match = pd.read_csv(match_file, encoding='ISO-8859-1')

    df_analytical_viable = df_analytical[df_analytical['num_viable_and_used_ht'] > 0]

    df_match = df_match[~df_match['PTR_PRIME_OPO_REFUSAL_ID'].between(720, 739)]
    df_match = df_match[~df_match['PTR_PRIME_OPO_REFUSAL_ID'].between(800, 819)]

    df_matched = df_match[df_match['PX_ID'].isin(df_analytical_viable['PX_ID'])]

    df_matched = df_matched.merge(df_refusal[['CD', 'Description']], left_on='PTR_PRIME_OPO_REFUSAL_ID', right_on='CD', how='left')

    refusal_counts = df_matched['PTR_PRIME_OPO_REFUSAL_ID'].value_counts().reset_index()
    refusal_counts.columns = ['Refusal Code', 'Number']

    total = refusal_counts['Number'].sum()
    refusal_counts['Percentages'] = (refusal_counts['Number'] / total * 100).round(2)

    refusal_counts.to_csv(output_file, index=False)
    print(f"Figure 1 data saved to {output_file}")

    plt.figure(figsize=(18, 14))  # Increased figure size to fit all labels
    plt.pie(refusal_counts['Percentages'], labels=refusal_counts['Refusal Code'], autopct='%1.1f%%', startangle=140)
    plt.title('Refusal Reasons Distribution (by Code)', fontsize=18)
    plt.tight_layout(pad=4)  # Additional padding
    plt.savefig('figure1_piechart.png')
    plt.show()

    refusal_counts['Refusal Code'] = refusal_counts['Refusal Code'].astype(str)  # Ensure x-values are strings
    plt.figure(figsize=(22, 14))  # Further increased figure size for all codes
    plt.bar(range(len(refusal_counts['Refusal Code'])), refusal_counts['Number'], color='skyblue')

    plt.xticks(range(len(refusal_counts['Refusal Code'])), refusal_counts['Refusal Code'], rotation=90, ha='right', fontsize=14)
    plt.xlabel('Refusal Reason Code', fontsize=16)
    plt.ylabel('Number of Refusals', fontsize=16)
    plt.title('Refusal Reasons (by Code)', fontsize=18)
    plt.tight_layout(pad=4)  # Additional padding to prevent text cutoff
    plt.savefig('figure1_bargraph.png')
    plt.show()

def calculate_median_age_with_iqr(match_file):
    """
    Calculate and print the median age and IQR (Interquartile Range) for DON_AGE
    when both viable_offer and used_ht are 1 in match2021and2022condensed.csv,
    excluding rows with PTR_PRIME_OPO_REFUSAL_ID between 720-739 and 800-819.
    """
    df_match = pd.read_csv(match_file, encoding='ISO-8859-1')

    filtered_df = df_match[(df_match['viable_offer'] == 1) & (df_match['used_ht'] == 1)]

    exclusion_filter = ~(
        ((filtered_df['PTR_PRIME_OPO_REFUSAL_ID'] >= 720) & (filtered_df['PTR_PRIME_OPO_REFUSAL_ID'] <= 739)) |
        ((filtered_df['PTR_PRIME_OPO_REFUSAL_ID'] >= 800) & (filtered_df['PTR_PRIME_OPO_REFUSAL_ID'] <= 819))
    )
    filtered_df = filtered_df[exclusion_filter]

    # Calculate the median age and IQR
    median_age = filtered_df['DON_AGE'].median()
    q1 = filtered_df['DON_AGE'].quantile(0.25)
    q3 = filtered_df['DON_AGE'].quantile(0.75)
    iqr = q3 - q1

    print(f"Median Age: {median_age:.2f}")
    print(f"25th Percentile (Q1): {q1:.2f}")
    print(f"75th Percentile (Q3): {q3:.2f}")
    print(f"Interquartile Range (IQR): {iqr:.2f}")


# --------------------------------------------------------------------------------------------------------------------
#  --------------------------------------------INDEPENDENT PROJECT START----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
def process_match_and_patients(match_file, analytic_file, active_dates_file, output_file):
    df_match = pd.read_csv(match_file, encoding='utf-8')
    df_analytic = pd.read_csv(analytic_file, encoding='utf-8')
    df_active_dates = pd.read_csv(active_dates_file, encoding='utf-8')

    df_analytic = df_analytic.merge(df_active_dates, on='PX_ID', how='left')

    df_analytic['CAN_INIT_ACT_STAT_DT'] = pd.to_datetime(df_analytic['CAN_INIT_ACT_STAT_DT'], errors='coerce')
    df_analytic = df_analytic[
        df_analytic['CAN_INIT_ACT_STAT_DT'].notna() & (df_analytic['CAN_INIT_ACT_STAT_DT'].dt.year == 2021)]

    # Filter only rows where age_listing >= 18 in both datasets
    df_match = df_match[df_match['age_listing'] >= 18]
    df_analytic = df_analytic[df_analytic['age_listing'] >= 18]

    df_match.to_csv("IP_match.csv", index=False)
    print("Filtered match file saved as IP_match.csv")

    df_match = pd.read_csv("IP_match.csv", encoding='utf-8')

    # Convert MATCH_SUBMIT_DT and the necessary columns in df_analytic to datetime
    '''
    df_match['MATCH_SUBMIT_DT'] = pd.to_datetime(df_match['MATCH_SUBMIT_DT'], errors='coerce',
                                                 format='%d%b%Y:%H:%M:%S.%f')
    df_analytic['CAN_INIT_ACT_STAT_DT'] = pd.to_datetime(df_analytic['CAN_INIT_ACT_STAT_DT'], errors='coerce')
    df_analytic['CAN_INIT_INACT_STAT_DT'] = pd.to_datetime(df_analytic['CAN_INIT_INACT_STAT_DT'], errors='coerce')
    df_analytic['rem_dt'] = pd.to_datetime(df_analytic['rem_dt'], errors='coerce')
    '''
    # ^ Replaced later with a separate function

    # Replace CAN_INIT_INACT_STAT_DT with rem_dt if it is blank
    df_analytic['CAN_INIT_INACT_STAT_DT'] = df_analytic['CAN_INIT_INACT_STAT_DT'].fillna(df_analytic['rem_dt'])

    # Ensure CAN_INIT_INACT_STAT_DT is not before CAN_INIT_ACT_STAT_DT, and handle cases where both are blank
    df_analytic['end_date'] = df_analytic.apply(
        lambda row: (
            row['CAN_INIT_ACT_STAT_DT'] + pd.DateOffset(days=365) if pd.isna(row['CAN_INIT_INACT_STAT_DT']) and pd.isna(
                row['rem_dt'])
            else min(
                row['CAN_INIT_ACT_STAT_DT'] + pd.DateOffset(days=365),
                row['rem_dt'] if pd.notna(row['CAN_INIT_INACT_STAT_DT']) and row['CAN_INIT_INACT_STAT_DT'] < row[
                    'CAN_INIT_ACT_STAT_DT']
                else row['CAN_INIT_INACT_STAT_DT']
            )
        ), axis=1
    )

    # mm/dd/yy format
    df_analytic['end_date'] = pd.to_datetime(df_analytic['end_date']).dt.strftime('%m/%d/%y')

    # time active calculations
    df_analytic['time_active'] = (pd.to_datetime(df_analytic['end_date']) - df_analytic['CAN_INIT_ACT_STAT_DT']).dt.days
    df_analytic.loc[df_analytic['time_active'] > 365, 'time_active'] = 365

    def calculate_offers(px_id):
        offers = df_match[df_match['PX_ID'] == px_id]
        active_start = df_analytic.loc[df_analytic['PX_ID'] == px_id, 'CAN_INIT_ACT_STAT_DT'].values[0]
        end_date = pd.to_datetime(df_analytic.loc[df_analytic['PX_ID'] == px_id, 'end_date']).values[0]
        offers = offers[(offers['MATCH_SUBMIT_DT'] >= active_start) & (offers['MATCH_SUBMIT_DT'] <= end_date)]
        return len(offers)

    def calculate_viable_offers(px_id):
        offers = df_match[df_match['PX_ID'] == px_id]
        active_start = df_analytic.loc[df_analytic['PX_ID'] == px_id, 'CAN_INIT_ACT_STAT_DT'].values[0]
        end_date = pd.to_datetime(df_analytic.loc[df_analytic['PX_ID'] == px_id, 'end_date']).values[0]
        offers = offers[(offers['MATCH_SUBMIT_DT'] >= active_start) & (offers['MATCH_SUBMIT_DT'] <= end_date) & (
                    offers['viable_offer'] > 0)]
        return len(offers)

    def calculate_viable_ht_offers(px_id):
        offers = df_match[df_match['PX_ID'] == px_id]
        active_start = df_analytic.loc[df_analytic['PX_ID'] == px_id, 'CAN_INIT_ACT_STAT_DT'].values[0]
        end_date = pd.to_datetime(df_analytic.loc[df_analytic['PX_ID'] == px_id, 'end_date']).values[0]
        offers = offers[(offers['MATCH_SUBMIT_DT'] >= active_start) & (offers['MATCH_SUBMIT_DT'] <= end_date) & (
                    offers['viable_offer'] > 0) & (offers['used_ht'] > 0)]
        return len(offers)

    def calculate_first_offer(px_id):
        offers = df_match[df_match['PX_ID'] == px_id]
        active_start = df_analytic.loc[df_analytic['PX_ID'] == px_id, 'CAN_INIT_ACT_STAT_DT'].values[0]
        end_date = pd.to_datetime(df_analytic.loc[df_analytic['PX_ID'] == px_id, 'end_date']).values[0]
        valid_offers = offers[(offers['MATCH_SUBMIT_DT'] >= active_start) & (offers['MATCH_SUBMIT_DT'] <= end_date)]
        if not valid_offers.empty:
            first_offer_date = valid_offers['MATCH_SUBMIT_DT'].min()
            return (first_offer_date - active_start).days
        return None

    def calculate_accept_date(px_id):
        offers = df_match[df_match['PX_ID'] == px_id]
        accepted_offers = offers[offers['PTR_OFFER_ACPT'] == 'Y']
        if not accepted_offers.empty:
            return accepted_offers['MATCH_SUBMIT_DT'].min()
        return None

    def calculate_accepted_offer(px_id):
        offers = df_match[df_match['PX_ID'] == px_id]
        return int((offers['PTR_OFFER_ACPT'] == 'Y').any())

    def calculate_accepted_offer_before_end_date(px_id):
        accept_date = pd.to_datetime(df_analytic.loc[df_analytic['PX_ID'] == px_id, 'accept_date']).values[0]
        end_date = pd.to_datetime(df_analytic.loc[df_analytic['PX_ID'] == px_id, 'end_date']).values[0]
        return int(pd.notna(accept_date) and accept_date <= end_date)

    df_analytic['num_offers'] = df_analytic['PX_ID'].apply(calculate_offers)
    df_analytic['num_offers_viable'] = df_analytic['PX_ID'].apply(calculate_viable_offers)
    df_analytic['num_offers_viable_HT'] = df_analytic['PX_ID'].apply(calculate_viable_ht_offers)
    df_analytic['first_offer'] = df_analytic['PX_ID'].apply(calculate_first_offer)
    df_analytic['accept_date'] = df_analytic['PX_ID'].apply(calculate_accept_date)
    df_analytic['accepted_offer'] = df_analytic['PX_ID'].apply(calculate_accepted_offer)
    df_analytic['accepted_offer_before_end_date'] = df_analytic['PX_ID'].apply(calculate_accepted_offer_before_end_date)

    # ccept_date as mm/dd/yy
    df_analytic['accept_date'] = pd.to_datetime(df_analytic['accept_date']).dt.strftime('%m/%d/%y')

    df_analytic.to_csv(output_file, index=False)
    print(f"IP_analytical_cohort saved to {output_file}")

def summarize_analytical_cohort(input_file, output_file):
    df = pd.read_csv(input_file) # IP_analytical_cohort.csv file

    summary_data = {}

    def get_summary_stats(column):
        return {
            '% of total': round((df[column] > 0).mean() * 100, 2),
            '#': (df[column] > 0).sum(),
            'mean': round(df[column].mean(), 2),
            'median': round(df[column].median(), 2),
            'IQR': round(df[column].quantile(0.75) - df[column].quantile(0.25), 2),
            'max': df[column].max(),
            'min': df[column].min()
        }

    summary_data['any_offer'] = get_summary_stats('num_offers')
    summary_data['any_viable_offer'] = get_summary_stats('num_offers_viable')
    summary_data['viable_and_used_for_HT'] = get_summary_stats('num_offers_viable_HT')
    summary_data['accepted_offer_before_end_date'] = get_summary_stats('accepted_offer_before_end_date')

    summary_df = pd.DataFrame(summary_data).T
    summary_df.reset_index(inplace=True)
    summary_df.columns = ['Variable', '% of total', '#', 'mean', 'median', 'IQR', 'max', 'min']

    summary_df.to_csv(output_file, index=False)
    print(f"Data summary saved to {output_file}")

    variables = {
        'num_offers': 'Any Offer',
        'num_offers_viable': 'Any Viable Offer',
        'num_offers_viable_HT': 'Viable and Used for HT',
        'accepted_offer_before_end_date': 'Accepted Offer Before End Date'
    }

    for column, title in variables.items():
        plt.figure(figsize=(10, 6))
        plt.hist(df[column], bins=30, edgecolor='black')
        plt.title(f'Histogram of {title}')
        plt.xlabel(title)
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(f'{title.replace(" ", "_").lower()}_histogram.png')
        plt.show()

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.scatter(df['time_active'], df['num_offers'], color='blue', label='num_offers', alpha=0.6)
    plt.xlabel("Time Active (days)")
    plt.ylabel("Number of Offers")
    plt.title("Time Active vs. Number of Offers")
    plt.legend()
    plt.savefig("time_active_vs_num_offers.png")
    plt.close()

    plt.subplot(3, 1, 2)
    plt.scatter(df['time_active'], df['num_offers_viable'], color='green', label='num_offers_viable', alpha=0.6)
    plt.xlabel("Time Active (days)")
    plt.ylabel("Number of Viable Offers")
    plt.title("Time Active vs. Number of Viable Offers")
    plt.legend()
    plt.savefig("time_active_vs_num_offers_viable.png")
    plt.close()

    plt.subplot(3, 1, 3)
    plt.scatter(df['time_active'], df['num_offers_viable_HT'], color='red', label='num_offers_viable_HT', alpha=0.6)
    plt.xlabel("Time Active (days)")
    plt.ylabel("Number of Viable HT Offers")
    plt.title("Time Active vs. Number of Viable HT Offers")
    plt.legend()
    plt.savefig("time_active_vs_num_offers_viable_HT.png")
    plt.close()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(df['time_active'], df['num_offers'], color='blue', label='num_offers', alpha=0.6)
    plt.scatter(df['time_active'], df['num_offers_viable'], color='green', label='num_offers_viable', alpha=0.6)
    plt.scatter(df['time_active'], df['num_offers_viable_HT'], color='red', label='num_offers_viable_HT', alpha=0.6)
    plt.xlabel("Time Active (days)")
    plt.ylabel("Number of Offers")
    plt.title("Time Active vs. Offers Comparison")
    plt.legend()
    plt.savefig("time_active_vs_offers_comparison.png")
    plt.close()
    plt.show()


def process_analytical_data(cohort_file, match_file, summarize_file, output_summary_file, output_table_file):
    df_cohort = pd.read_csv(cohort_file, encoding='utf-8')
    df_match = pd.read_csv(match_file, encoding='utf-8')
    df_summarize = pd.read_csv(summarize_file, encoding='utf-8')

    df_match['MATCH_SUBMIT_DT'] = pd.to_datetime(df_match['MATCH_SUBMIT_DT'], format='%d%b%Y:%H:%M:%S.%f', errors='coerce')
    df_match['MATCH_SUBMIT_DT'] = df_match['MATCH_SUBMIT_DT'].dt.strftime('%m/%d/%y')

    df_cohort['time_to_first_viable_offer_used_HT'] = df_cohort['PX_ID'].apply(
        lambda px_id: calculate_time_to_first_viable_offer_used_HT(px_id, df_match, df_cohort)
    )

    update_summary_with_new_variable(df_cohort, 'time_to_first_viable_offer_used_HT', df_summarize)

    df_cohort['rate_of_viable_offers_used_HT'] = df_cohort['num_offers_viable_HT'] / df_cohort['time_active']

    df_cohort.to_csv("IP_analytical_cohort_updated.csv", index=False)

    df_summarize.to_csv(output_summary_file, index=False)
    print("Updated analytical cohort saved as IP_analytical_cohort_updated.csv")
    print(f"Saved summary data to {output_summary_file} and analytical cohort table to {output_table_file}.")


def calculate_time_to_first_viable_offer_used_HT(px_id, df_match, df_cohort):

    cohort_row = df_cohort[df_cohort['PX_ID'] == px_id]
    if cohort_row.empty:
        return np.nan

    act_stat_date = pd.to_datetime(cohort_row['CAN_INIT_ACT_STAT_DT'].values[0], errors='coerce')
    if pd.isna(act_stat_date):
        return np.nan

    viable_used_ht_offers = df_match[(df_match['PX_ID'] == px_id) &
                                     (df_match['viable_offer'] == 1) &
                                     (df_match['used_ht'] == 1)].copy()

    viable_used_ht_offers['MATCH_SUBMIT_DT'] = pd.to_datetime(
        viable_used_ht_offers['MATCH_SUBMIT_DT'], format='%d%b%Y:%H:%M:%S.%f', errors='coerce'
    )

    viable_used_ht_offers = viable_used_ht_offers.dropna(subset=['MATCH_SUBMIT_DT'])

    viable_used_ht_offers = viable_used_ht_offers[viable_used_ht_offers['MATCH_SUBMIT_DT'] > act_stat_date]

    if viable_used_ht_offers.empty:
        return np.nan

    closest_date = viable_used_ht_offers['MATCH_SUBMIT_DT'].min()
    time_diff = (closest_date - act_stat_date).days

    return time_diff


def update_summary_with_new_variable(df, variable, df_summary):
    data = df[variable].dropna()
    summary_stats = {
        'variable': variable,
        '#': len(data),
        'mean': data.mean(),
        'median': data.median(),
        'IQR': data.quantile(0.75) - data.quantile(0.25),
        'max': data.max(),
        'min': data.min()
    }
    df_summary = pd.concat([df_summary, pd.DataFrame([summary_stats])], ignore_index=True)

# Creating table for Independent Study project presentation
def generate_table_data(df):
    categories = {
        'total': lambda df: df,
        'female': lambda df: df[df['female'] == 1],
        'male': lambda df: df[df['female'] == 0],
        'typeA': lambda df: df[df['typeA'] == 1],
        'typeB': lambda df: df[df['typeB'] == 1],
        'typeAB': lambda df: df[df['typeAB'] == 1],
        'typeO': lambda df: df[df['typeO'] == 1],
        'white': lambda df: df[df['white'] == 1],
        'black': lambda df: df[df['black'] == 1],
        'asian': lambda df: df[df['asian'] == 1],
        'hispanic_latino': lambda df: df[df['hispanic_latino'] == 1],
        'other_race': lambda df: df[df['other_race'] == 1],
        'age (<40)': lambda df: df[df['age_listing'] < 40],
        'age (40-59)': lambda df: df[(df['age_listing'] >= 40) & (df['age_listing'] < 60)],
        'age (60+)': lambda df: df[df['age_listing'] >= 60],
        'bmi (<20)': lambda df: df[df['bmi'] < 20],
        'bmi (20-29)': lambda df: df[(df['bmi'] >= 20) & (df['bmi'] <= 29)],
        'bmi (30+)': lambda df: df[df['bmi'] >= 30],
        'nidcm': lambda df: df[df['nidcm'] == 1],
        'ischemic': lambda df: df[df['ischemic'] == 1],
        'rcm': lambda df: df[df['rcm'] == 1],
        'hcm': lambda df: df[df['hcm'] == 1],
        'congenital': lambda df: df[df['congenital'] == 1],
        'valvular': lambda df: df[df['valvular'] == 1],
        'retx_gf': lambda df: df[df['retx_gf'] == 1],
        'other_etiology': lambda df: df[df['other_etiology'] == 1],
        'no_college': lambda df: df[df['no_college'] == 1],
        'edu_unknown': lambda df: df[df['edu_unknown'] == 1],
        'college': lambda df: df[df['college'] == 1],
        'private_ins': lambda df: df[df['private_ins'] == 1],
        'medicaid': lambda df: df[df['medicaid'] == 1],
        'medicare': lambda df: df[df['medicare'] == 1],
        'other_pay': lambda df: df[df['other_pay'] == 1],
        'missing_pay': lambda df: df[df['missing_pay'] == 1],
        'othergovt_ins': lambda df: df[df['othergovt_ins'] == 1],
        'diabetes': lambda df: df[df['diabetes'] == 1],
        'dial_listing': lambda df: df[df['dial_listing'] == 1],
        'smoking': lambda df: df[df['smoking'] == 1],
        'initstat_new1': lambda df: df[df['initstat_new1'] == 1],
        'initstat_new2': lambda df: df[df['initstat_new2'] == 1],
        'initstat_new3': lambda df: df[df['initstat_new3'] == 1],
        'initstat_new4': lambda df: df[df['initstat_new4'] == 1],
        'initstat_new5': lambda df: df[df['initstat_new5'] == 1],
        'initstat_new6': lambda df: df[df['initstat_new6'] == 1]
    }

    columns = {
        'received offer': lambda df: (df['num_offers'] >= 1),
        'received viable offer': lambda df: (df['num_offers_viable'] >= 1),
        'received viable offer HT': lambda df: (df['num_offers_viable_HT'] >= 1),
        'accepted offer': lambda df: (df['accepted_offer'] >= 1),
        'accepted offer before end date': lambda df: (df['accepted_offer_before_end_date'] >= 1)
    }

    data = []
    for row_name, row_func in categories.items():
        row_data = {'Category': row_name}
        for col_name, col_func in columns.items():
            col_values = row_func(df)
            count = col_func(col_values).sum()
            percentage = (count / len(df)) * 100 if len(df) > 0 else 0
            row_data[f'{col_name} #'] = count
            row_data[f'{col_name} %'] = round(percentage, 2)
        data.append(row_data)

    return data

# Craeting the table based on the data
def generate_summary_table(input_file, output_file):
    df = pd.read_csv(input_file, encoding='utf-8')

    categories = {
        'female': df['female'] == 1,
        'male': df['female'] == 0,
        'typeA': df['typeA'] == 1,
        'typeB': df['typeB'] == 1,
        'typeAB': df['typeAB'] == 1,
        'typeO': df['typeO'] == 1,
        'white': df['white'] == 1,
        'black': df['black'] == 1,
        'asian': df['asian'] == 1,
        'hispanic_latino': df['hispanic_latino'] == 1,
        'other_race': df['other_race'] == 1,
        'age (<40)': df['age_listing'] < 40,
        'age (40-59)': (df['age_listing'] >= 40) & (df['age_listing'] < 60),
        'age (60+)': df['age_listing'] >= 60,
        'bmi (<20)': df['bmi'] < 20,
        'bmi (20-29)': (df['bmi'] >= 20) & (df['bmi'] <= 29),
        'bmi (30+)': df['bmi'] >= 30,
    }

    columns = {
        'Total number': lambda sub_df: len(sub_df),
        'Received offer #': lambda sub_df: (sub_df['num_offers'] >= 1).sum(),
        'Received offer %': lambda sub_df: ((sub_df['num_offers'] >= 1).sum() / len(sub_df) * 100) if len(sub_df) > 0 else 0,
        'Received viable offer #': lambda sub_df: (sub_df['num_offers_viable'] >= 1).sum(),
        'Received viable offer %': lambda sub_df: ((sub_df['num_offers_viable'] >= 1).sum() / len(sub_df) * 100) if len(sub_df) > 0 else 0,
        'Received viable offer HT #': lambda sub_df: (sub_df['num_offers_viable_HT'] >= 1).sum(),
        'Received viable offer HT %': lambda sub_df: ((sub_df['num_offers_viable_HT'] >= 1).sum() / len(sub_df) * 100) if len(sub_df) > 0 else 0,
        'Accepted offer #': lambda sub_df: (sub_df['accepted_offer'] >= 1).sum(),
        'Accepted offer %': lambda sub_df: ((sub_df['accepted_offer'] >= 1).sum() / len(sub_df) * 100) if len(sub_df) > 0 else 0,
        'Accepted offer before end date #': lambda sub_df: (sub_df['accepted_offer_before_end_date'] >= 1).sum(),
        'Accepted offer before end date %': lambda sub_df: ((sub_df['accepted_offer_before_end_date'] >= 1).sum() / len(sub_df) * 100) if len(sub_df) > 0 else 0,
    }

    summary_data = []

    total_row = {'Category': 'Total'}
    for col_name, col_func in columns.items():
        total_row[col_name] = col_func(df)
    summary_data.append(total_row)

    for category_name, category_filter in categories.items():
        category_data = df[category_filter]
        row = {'Category': category_name}
        for col_name, col_func in columns.items():
            row[col_name] = col_func(category_data)
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    p_values_viable_offer_ht = []
    p_values_accepted_offer = []

    total_viable_offer_ht = total_row['Received viable offer HT #']
    total_not_viable_offer_ht = total_row['Total number'] - total_viable_offer_ht

    total_accepted_offer = total_row['Accepted offer before end date #']
    total_not_accepted_offer = total_row['Total number'] - total_accepted_offer

    for _, row in summary_df.iterrows():
        if row['Category'] == 'Total':
            p_values_viable_offer_ht.append(None)
            p_values_accepted_offer.append(None)
        else:
            observed_viable = row['Received viable offer HT #']
            observed_not_viable = row['Total number'] - observed_viable
            expected_viable = total_viable_offer_ht * (row['Total number'] / total_row['Total number'])
            expected_not_viable = total_not_viable_offer_ht * (row['Total number'] / total_row['Total number'])
            chi_sq_viable = ((observed_viable - expected_viable) ** 2 / expected_viable) + \
                            ((observed_not_viable - expected_not_viable) ** 2 / expected_not_viable)

            observed_accepted = row['Accepted offer before end date #']
            observed_not_accepted = row['Total number'] - observed_accepted
            expected_accepted = total_accepted_offer * (row['Total number'] / total_row['Total number'])
            expected_not_accepted = total_not_accepted_offer * (row['Total number'] / total_row['Total number'])
            chi_sq_accepted = ((observed_accepted - expected_accepted) ** 2 / expected_accepted) + \
                              ((observed_not_accepted - expected_not_accepted) ** 2 / expected_not_accepted)

            p_values_viable_offer_ht.append(round(chi_square_cdf(chi_sq_viable, 1), 4))
            p_values_accepted_offer.append(round(chi_square_cdf(chi_sq_accepted, 1), 4))

    summary_df['p-value (viable offer HT)'] = p_values_viable_offer_ht
    summary_df['p-value (accepted offer)'] = p_values_accepted_offer

    summary_df.to_csv(output_file, index=False)
    print(f"Summary table with totals and p-values saved to {output_file}.")

def chi_square_cdf(chi_squared, df):
    from math import exp, gamma
    k = df / 2
    x = chi_squared / 2

    # Incomplete gamma function approximation
    t = 1.0
    s = t
    for i in range(1, 101):
        t *= x / (k + i)
        s += t
    return (exp(-x) * x ** k * s / gamma(k))

# THIS FUNCTION MIXES THE DATE ERRORS
def update_match_submit_dt(input_file, output_file):
    """
    Update MATCH_SUBMIT_DT column in IP_match.csv to the format 'm/d/yyyy'.
    Example: Change '25DEC2020:10:24:43.797' to '4/15/2020'.
    """
    df_match = pd.read_csv(input_file, encoding='utf-8')

    df_match['MATCH_SUBMIT_DT'] = pd.to_datetime(
        df_match['MATCH_SUBMIT_DT'], format='%d%b%Y:%H:%M:%S.%f', errors='coerce'
    ).dt.strftime('%-m/%-d/%Y')

    df_match.to_csv(output_file, index=False)
    print(f"Updated MATCH_SUBMIT_DT and saved to {output_file}")


def calculate_time_to_first_viable_offer(input_cohort_file, input_match_file, output_cohort_file):

    df_cohort = pd.read_csv(input_cohort_file, encoding='utf-8')
    df_match = pd.read_csv(input_match_file, encoding='utf-8')

    df_cohort['CAN_INIT_ACT_STAT_DT'] = pd.to_datetime(df_cohort['CAN_INIT_ACT_STAT_DT'], errors='coerce')
    df_match['MATCH_SUBMIT_DT'] = pd.to_datetime(df_match['MATCH_SUBMIT_DT'], errors='coerce')

    df_cohort['time_to_first_viable_offer_used_HT'] = None

    for index, row in df_cohort.iterrows():
        px_id = row['PX_ID']
        act_stat_date = row['CAN_INIT_ACT_STAT_DT']

        if pd.isna(act_stat_date):
            continue  # Skip if activation date is missing

        viable_offers = df_match[
            (df_match['PX_ID'] == px_id) &
            (df_match['viable_offer'] == 1) &
            (df_match['used_ht'] == 1)
        ]

        if not viable_offers.empty:
            earliest_offer_date = viable_offers['MATCH_SUBMIT_DT'].min()
            time_difference = (earliest_offer_date - act_stat_date).days
            df_cohort.at[index, 'time_to_first_viable_offer_used_HT'] = time_difference

    df_cohort.to_csv(output_cohort_file, index=False)
    print(f"Updated analytical cohort saved to {output_cohort_file}")


def create_summary_bar_charts(input_summary_file):

    df_summary = pd.read_csv(input_summary_file)

    variable_groups = {
        'male_vs_female': ['male', 'female'],
        'blood_types': ['typeA', 'typeB', 'typeAB', 'typeO'],
        'ethnicities': ['white', 'black', 'asian', 'hispanic_latino', 'other_race'],
        'age_groups': ['age (<40)', 'age (40-59)', 'age (60+)'],
        'bmi_groups': ['bmi (<20)', 'bmi (20-29)', 'bmi (30+)'],
        'initstat_categories': ['initstat_new1', 'initstat_new2', 'initstat_new3', 'initstat_new4', 'initstat_new5', 'initstat_new6']
    }

    color_schemes = {
        'male_vs_female': ['#9370DB', '#BA55D3'],
        'blood_types': ['#9370DB', '#BA55D3', '#8A2BE2', '#DDA0DD'],
        'ethnicities': ['#9370DB', '#BA55D3', '#8A2BE2', '#DDA0DD', '#E6E6FA'],
        'age_groups': ['#9370DB', '#BA55D3', '#8A2BE2'],
        'bmi_groups': ['#9370DB', '#BA55D3', '#8A2BE2'],
        'initstat_categories': ['#9370DB', '#BA55D3', '#8A2BE2', '#DDA0DD', '#E6E6FA', '#FFB6C1']
    }

    for group_name, variables in variable_groups.items():
        received_offer_percents = []
        accepted_offer_percents = []

        for var in variables:
            received_offer_percents.append(round(df_summary.loc[df_summary['Category'] == var, 'Received viable offer HT %'].values[0], 1))
            accepted_offer_percents.append(round(df_summary.loc[df_summary['Category'] == var, 'Accepted offer before end date %'].values[0], 1))

        x = range(len(variables))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar([i - width / 2 for i in x], received_offer_percents, width, label='Received Offer', color=color_schemes[group_name][0])
        bars2 = ax.bar([i + width / 2 for i in x], accepted_offer_percents, width, label='Accepted Offer', color=color_schemes[group_name][1])

        for bar, percent in zip(bars1, received_offer_percents):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{percent}%', ha='center', va='bottom', fontsize=10)

        for bar, percent in zip(bars2, accepted_offer_percents):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{percent}%', ha='center', va='bottom', fontsize=10)

        ax.set_title(f'{group_name.replace("_", " ").title()} Summary', fontsize=14)
        ax.set_xlabel('Categories', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 100)  # Set Y-axis to range from 0 to 100
        ax.legend(loc='upper right')

        output_path = f'{group_name}_summary_chart.png'
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"Saved bar chart for {group_name} to {output_path}")


def main():
    while True:
        print("\nMenu:")
        print("1. Condense Master CSV file")
        print("2. Clean Up Offer Dataset")
        print("3. Create Analytical Cohort")
        print("4. Condense Cleaned Offer Dataset")
        print("5. Update Analytical Cohort")
        print("6. Analyze Viable Offers")
        print("7. Create Table 1")
        print("8. Creating Pie Charts")
        print("9. Creating Bar Charts")
        print("10. Logistic Regression Analysis and Plots")
        print("11. Updated Table 1 with chi-squared")
        print("12. Create Figure 1")
        print("13. Median DONOR age + IQR")
        print("------------------------------------------------")
        print("14. Independent Project: Data Cleanup")
        print("15. Summarize IP data")
        print("16. IP Table 1")
        print("17. Summary table")
        print("18. Update match")
        print("19. Update time to viable offer and HT")
        print("20. Bar charts for IP")
        print("0. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            file_path = "match2021and2022combined.csv"
            output_path = "match2021and2022truncated.csv"
            condense_csv(file_path, output_path)
        elif choice == '2':
            file_path_1 = "match2021and2022truncated.csv"
            file_path_2 = "SRTR analytic dataset_patients_021724.csv"
            file_path_3 = "donors.csv"
            output_path = "match2021and2022cleaned.csv"
            clean_up_offer_dataset(file_path_1, file_path_2, file_path_3, output_path)
        elif choice == '3':
            file_path = "SRTR analytic dataset_patients_021724.csv"
            output_path = "analyticalcohort.csv"
            create_analytical_cohort(file_path, output_path)
        elif choice == '4':
            file_path_1 = "match2021and2022cleaned.csv"
            file_path_2 = "analyticalcohort.csv"
            output_path = "match2021and2022condensed.csv"
            condense_cleaned_offer_dataset(file_path_1, file_path_2, output_path)
        elif choice == '5':
            file_path_1 = "analyticalcohort.csv"
            file_path_2 = "match2021and2022condensed.csv"
            output_path = "updated_analyticalcohort.csv"
            update_analytical_cohort(file_path_1, file_path_2, output_path)
        elif choice == '6':
            file_path = "updated_analyticalcohort.csv"
            analyze_viable_offers(file_path)
        elif choice == '7':
            file_path_1 = "updated_analyticalcohort.csv"
            file_path_2 = "match2021and2022condensed.csv"
            file_path_3 = "ctr.csv"
            file_path_4 = "institutions.csv"
            output_path = "table1.csv"
            create_table_1(file_path_1, file_path_2, file_path_3, file_path_4, output_path)
        elif choice == '8':
            file_path_1 = "table1.csv"
            create_graphs(file_path_1)
        elif choice == '9':
            file_path_2 = "updated_analyticalcohort.csv"
            create_bar_charts(file_path_2)
        elif choice == '10':
            file_path_1 = "table1.csv"
            file_path_2 = "updated_analyticalcohort.csv"
            output_prefix = "logistic_regression_analysis"
            create_function_10(file_path_1, file_path_2, output_prefix)
        elif choice == '11':
            file_path_1 = "updated_analyticalcohort.csv"
            output_prefix = "table1_update.csv"
            create_function_11(file_path_1, output_prefix)
        elif choice == '12':
            file_path_1 = "refusal codes_v2.csv"
            file_path_2 = "updated_analyticalcohort.csv"
            file_path_3 = "match2021and2022condensed.csv"
            output_prefix = "Figure_1.csv"
            create_function_12(file_path_1, file_path_2, file_path_3, output_prefix)
        elif choice == '13':
            calculate_median_age_with_iqr('match2021and2022condensed.csv')
        elif choice == '14':
            process_match_and_patients('match2021and2022cleaned.csv', 'SRTR analytic dataset_patients_021724.csv', 'active_dates.csv', 'IP_analytical_cohort.csv')
        elif choice == '15':
            summarize_analytical_cohort('IP_analytical_cohort.csv', 'IP_data_summarize.csv')
        elif choice == "16":
            process_analytical_data('IP_analytical_cohort.csv', 'IP_match.csv', 'IP_data_summarize.csv',
                                    'IP_data_summarize_updated.csv', 'IP_table_1.csv')
        elif choice == "17":
            generate_summary_table("IP_analytical_cohort_updated.csv", "IP_summary_table.csv")
        elif choice == "18":
            update_match_submit_dt('IP_match.csv', 'IP_match_updated.csv')
        elif choice == "19":
            calculate_time_to_first_viable_offer('IP_analytical_cohort_updated.csv', 'IP_match_updated.csv',
                                                 'IP_analytical_cohort_updated.csv')
        elif choice == "20":
            create_summary_bar_charts('IP_summary_table.csv')
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
