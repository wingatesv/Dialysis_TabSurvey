import pandas as pd
import numpy as np

def mixup(patient_ids, augmentation_params):
    mixup_df = pd.DataFrame()

    data = pd.read_csv('/content/drive/MyDrive/NTU Project/df_selected.csv')

    print('Mixup lambda: ', augmentation_params['mixup_lambda'])
    # For each pair of patient IDs, create mixup data and add it to mixup_df
    for i in range(0, len(patient_ids), 2):
        if i + 1 >= len(patient_ids):
            break
            
        patient_id1 = patient_ids[i]
        patient_id2 = patient_ids[i+1]

        patient_data1 = data[data['patient ID'] == patient_id1].copy()
        patient_data2 = data[data['patient ID'] == patient_id2].copy()

        # Determine the number of data points from patient1 and patient2 based on lmbda_fraction
        num_data_points1 = int(len(patient_data1) * augmentation_params['mixup_lambda'])
        num_data_points2 = len(patient_data1) - num_data_points1

        # Select the data points from patient1 and patient2
        patient_data1_mixup = patient_data1.iloc[:num_data_points1]
        patient_data2_mixup = patient_data2.iloc[-num_data_points2:]

        # Concatenate the selected data points to create the mixup data
        mixup_data = pd.concat([patient_data1_mixup, patient_data2_mixup])

        # Generate a new patient ID for the mixup data
        new_patient_id = 'mixup_' + str(np.random.randint(1e6))  # This generates a random integer between 0 and 1e6
        mixup_data['patient ID'] = new_patient_id

        mixup_df = pd.concat([mixup_df, mixup_data])

    return mixup_df

def cutmix(patient_ids, augmentation_params):
    cutmix_df = pd.DataFrame()
    data = pd.read_csv('/content/drive/MyDrive/NTU Project/df_selected.csv')

   
    print('Cutmix lambda: ', augmentation_params['cutmix_lambda'])
    # For each pair of patient IDs, create mixup data and add it to mixup_df
    for i in range(0, len(patient_ids), 2):
        if i + 1 >= len(patient_ids):
            break
            
        patient_id1 = patient_ids[i]
        patient_id2 = patient_ids[i+1]

        patient_data1 = data[data['patient ID'] == patient_id1].copy()
        patient_data2 = data[data['patient ID'] == patient_id2].copy()

        # Determine the number of data points from patient1 and patient2 based on lmbda_fraction
        num_data_points1 = int(len(patient_data1) *  augmentation_params['cutmix_lambda'])
        num_data_points2 = len(patient_data1) - num_data_points1

        # Calculate the start and end indices of the cut region
        start_idx = len(patient_data1) // 2 - num_data_points2 // 2
        end_idx = start_idx + num_data_points2

        # Select the data points from patient1 and patient2
        patient_data1_cutmix = pd.concat([patient_data1.iloc[:start_idx], patient_data2.iloc[start_idx:end_idx], patient_data1.iloc[end_idx:]])

        # Generate a new patient ID for the cutmix data
        new_patient_id = 'cutmix_' + str(np.random.randint(1e6))  # This generates a random integer between 0 and 1e6
        patient_data1_cutmix['patient ID'] = new_patient_id

        cutmix_df = pd.concat([cutmix_df, patient_data1_cutmix])

    return cutmix_df




def add_gaussian_noise(patient_ids, augmentation_params):

    noise_df = pd.DataFrame()
    data = pd.read_csv('/content/drive/MyDrive/NTU Project/df_selected.csv')

    print('Gaussian Noise Level: ', augmentation_params['gaussian_noise_level'])

    # For each patient ID, create noise data and add it to noise_df
    for patient_id in patient_ids:
        patient_data = data[data['patient ID'] == patient_id].copy()

        # Only apply Gaussian noise to selected columns
        noise_columns = ['b2-M', '255nm', '280nm', '310nm', 'current dehydration volume', 'hourly dehydration volume', 'transmembrane pressure']

        # Add Gaussian noise to each value in the selected columns
        for column in noise_columns:
            noise = np.random.normal(0, augmentation_params['gaussian_noise_level'], size=patient_data[column].shape)
            patient_data[column] += noise

        # Generate a new patient ID for the noisy data
        new_patient_id = 'noise_' + str(np.random.randint(1e6))  # This generates a random integer between 0 and 1e6
        patient_data['patient ID'] = new_patient_id

        noise_df = pd.concat([noise_df, patient_data])

    return noise_df



def add_random_jitter(patient_ids, augmentation_params):
    jitter_df = pd.DataFrame()
    data = pd.read_csv('/content/drive/MyDrive/NTU Project/df_selected.csv')

    print('Jitter Level: ', augmentation_params['jitter_level'])

    # For each patient ID, create jitter data and add it to jitter_df
    for patient_id in patient_ids:
        patient_data = data[data['patient ID'] == patient_id].copy()

        # Only apply random jitter to selected columns
        jitter_columns = ['b2-M', '255nm', '280nm', '310nm', 'current dehydration volume', 'hourly dehydration volume', 'transmembrane pressure']

        # Add random jitter to each value in the selected columns
        for column in jitter_columns:
            jitter = np.random.uniform(-augmentation_params['jitter_level'], augmentation_params['jitter_level'], size=patient_data[column].shape)
            patient_data[column] += jitter

        # Generate a new patient ID for the jitter data
        new_patient_id = 'jitter_' + str(np.random.randint(1e6))  # This generates a random integer between 0 and 1e6
        patient_data['patient ID'] = new_patient_id

        jitter_df = pd.concat([jitter_df, patient_data])

    return jitter_df

