'''
modify the mixup and cutmix so that fixed patient features like weights, age, and etc remains the same, follow the first patient
modify the gaussian noise and random jittering to only augment continuous features only, those discrete features should remain discrete
if got time, modify the algorithm to that can check these condition automatically before augmenting

mixup is wrongly implemented, should be taking the average of the cutout patient data

novelty and contribution: DL results, propose novel augmentation framework for this dataset
'''
import pandas as pd
import numpy as np

def filter_df(data_path, use_absorbance_only, use_personalized_only, target_variable):
  df = pd.read_csv(data_path)

  if use_absorbance_only:
      # filter out the df to only use the columns: Patient ID, collection time, BUN, 255nm, 280nm and 310nm
      columns_to_use = ['patient ID', 'collection time', target_variable, '255nm', '280nm', '310nm']

  elif use_personalized_only:
    columns_to_use = [ 'patient ID', 'collection time', target_variable, 
                      'NMWCO', 'membrane area', 'venous pressure', 'arterial flow velocity',
                      'hourly dehydration volume', 'target dehydration amount', 'current dehydration volume',
                      'dialysate ion concentration', 'transmembrane pressure', 'dialysate flow rate',
                      'ultrafiltration coefficient', 'dialysis day',
                      'age', 'systolic pressure', 'duration of dialysis', 'height', 'dry body weight']
    
  else:
      columns_to_use = [ 'patient ID', 'collection time', target_variable, '255nm', '280nm', '310nm', 
                        'NMWCO', 'membrane area', 'venous pressure', 'arterial flow velocity',
                        'hourly dehydration volume', 'target dehydration amount', 'current dehydration volume',
                        'dialysate ion concentration', 'transmembrane pressure', 'dialysate flow rate',
                        'ultrafiltration coefficient', 'dialysis day',
                        'age', 'systolic pressure', 'duration of dialysis', 'height', 'dry body weight']

  df = df[columns_to_use]

  return df

def filter_continuous_columns(data):
    continuous_columns = data.select_dtypes(include=['float64', 'int64']).columns
    return data[continuous_columns]
    
def identify_fixed_columns(data):
    # Set a threshold for considering a column as fixed
    fixed_threshold = 0.99

    fixed_columns = []
    for column in data.columns:
        unique_values = data[column].nunique()
        total_values = data[column].count()

        # Check if a column has a high percentage of the same value (above the threshold)
        if unique_values / total_values < fixed_threshold:
            fixed_columns.append(column)

    return fixed_columns
  
def mixup_old(patient_ids, data_path, use_absorbance_only, use_personalized_only, target_variable, augmentation_params):
    mixup_df = pd.DataFrame()

    data = filter_df(data_path, use_absorbance_only, use_personalized_only, target_variable)

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

        # Now do the reverse: take the first part of patient2's data and concatenate it with the second part of patient1's data
        patient_data1_mixup = patient_data1.iloc[num_data_points1:]
        patient_data2_mixup = patient_data2.iloc[:-num_data_points2]

        mixup_data2 = pd.concat([patient_data2_mixup, patient_data1_mixup])

        new_patient_id2 = 'mixup_' + str(np.random.randint(1e6))
        mixup_data2['patient ID'] = new_patient_id2

        mixup_df = pd.concat([mixup_df, mixup_data2])

    return mixup_df

def generate_mixup_data(patient_data1, patient_data2, augmentation_params, target_variable):

    # Determine the number of data points from patient1 and patient2 based on lmbda_fraction
    num_data_points1 = int(len(patient_data1) * augmentation_params['mixup_lambda'])
    num_data_points2 = len(patient_data1) - num_data_points1
    # Select the data points from patient1 and patient2
    patient_data1_mixup1 = patient_data1.iloc[:num_data_points1]
    patient_data1_mixup2 = patient_data1.iloc[-num_data_points2:]
    patient_data2_mixup = patient_data2.iloc[-num_data_points2:]

    # Reset the index for patient_data1_mixup2
    patient_data1_mixup2 = patient_data1_mixup2.reset_index(drop=True)

    # Mixup the data points from patient1 and patient2
    mixup_data = pd.DataFrame()

    # Select the numeric columns to average
    numeric_columns = ['collection time', target_variable, '255nm', '280nm', '310nm', 
                        'venous pressure', 'arterial flow velocity',
                        'current dehydration volume',
                        'transmembrane pressure']

    # Calculate the average for each numeric column
    mixup_data[numeric_columns] = (patient_data1_mixup2[numeric_columns].values + patient_data2_mixup[numeric_columns].values) / 2

    # Fill NaN values in mixup_data with corresponding values from patient_data1_mixup2
    mixup_data = mixup_data.fillna(patient_data1_mixup2)

    mixup_data = pd.concat([patient_data1_mixup1, mixup_data], axis=0)

    # Fill NaN values in mixup_data with corresponding values from patient_data1
    mixup_data = mixup_data.fillna(patient_data1.reset_index(drop=True))

    mixup_data = mixup_data.reset_index(drop=True)

    # Generate a new patient ID for the mixup data
    new_patient_id = 'mixup_' + str(np.random.randint(1e6))
    mixup_data['patient ID'] = new_patient_id

    return mixup_data

def mixup(patient_ids, data_path, use_absorbance_only, use_personalized_only, target_variable, augmentation_params):
    mixup_df = pd.DataFrame()

    data = filter_df(data_path, use_absorbance_only, use_personalized_only, target_variable)

    print('Mixup lambda: ', augmentation_params['mixup_lambda'])
    # For each pair of patient IDs, create mixup data and add it to mixup_df
    for i in range(0, len(patient_ids), 2):
        if i + 1 >= len(patient_ids):
            break
            
        patient_id1 = patient_ids[i]
        patient_id2 = patient_ids[i+1]

        patient_data1 = data[data['patient ID'] == patient_id1].copy()
        patient_data2 = data[data['patient ID'] == patient_id2].copy()

        mixup_data_1 = generate_mixup_data(patient_data1, patient_data2, augmentation_params, target_variable)

        mixup_data_2 = generate_mixup_data(patient_data2, patient_data1, augmentation_params, target_variable)


        mixup_df = pd.concat([mixup_df, mixup_data_1, mixup_data_2])


    return mixup_df



def cutmix(patient_ids, data_path, use_absorbance_only, use_personalized_only, target_variable, augmentation_params):
    cutmix_df = pd.DataFrame()
    data = filter_df(data_path, use_absorbance_only, use_personalized_only, target_variable)

   
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

        # Now do the reverse: take the first part of patient2's data and concatenate it with the second part of patient1's data
        patient_data2_cutmix = pd.concat([patient_data2.iloc[:start_idx], patient_data1.iloc[start_idx:end_idx], patient_data2.iloc[end_idx:]])

        new_patient_id2 = 'cutmix_' + str(np.random.randint(1e6))
        patient_data2_cutmix['patient ID'] = new_patient_id2

        cutmix_df = pd.concat([cutmix_df, patient_data2_cutmix])

    return cutmix_df





def add_gaussian_noise(patient_ids, data_path, use_absorbance_only, use_personalized_only, target_variable, augmentation_params):

    noise_df = pd.DataFrame()
    data = filter_df(data_path, use_absorbance_only, use_personalized_only, target_variable)

    print('Gaussian Noise Level: ', augmentation_params['gaussian_noise_level'])

    # For each patient ID, create noise data and add it to noise_df
    for patient_id in patient_ids:
        patient_data = data[data['patient ID'] == patient_id].copy()

        # Only apply Gaussian noise to selected columns

        excluded_columns = [target_variable, 'patient ID', 'collection time']
        noise_columns = [col for col in data.columns if col not in excluded_columns]


        # Add Gaussian noise to each value in the selected columns
        for column in noise_columns:
            noise = np.random.normal(0, augmentation_params['gaussian_noise_level'], size=patient_data[column].shape)
            patient_data[column] += noise

        # Generate a new patient ID for the noisy data
        new_patient_id = 'noise_' + str(np.random.randint(1e6))  # This generates a random integer between 0 and 1e6
        patient_data['patient ID'] = new_patient_id

        noise_df = pd.concat([noise_df, patient_data])

    return noise_df


# def add_gaussian_noise(patient_ids, data_path, use_absorbance_only, use_personalized_only, target_variable, augmentation_params):

#     noise_df = pd.DataFrame()
#     data = filter_df(data_path, use_absorbance_only, use_personalized_only, target_variable)

#     print('Gaussian Noise Level: ', augmentation_params['gaussian_noise_level'])

#     # Filter only continuous columns
#     continuous_data = filter_continuous_columns(data)

#     # For each patient ID, create noise data and add it to noise_df
#     for patient_id in patient_ids:
#         patient_data = continuous_data[continuous_data['patient ID'] == patient_id].copy()

#         # Only apply Gaussian noise to selected columns
#         excluded_columns = [target_variable, 'patient ID', 'collection time']
#         noise_columns = [col for col in patient_data.columns if col not in excluded_columns]

#         # Add Gaussian noise to each value in the selected columns
#         for column in noise_columns:
#             noise = np.random.normal(0, augmentation_params['gaussian_noise_level'], size=patient_data[column].shape)
#             patient_data[column] += noise

#         # Generate a new patient ID for the noisy data
#         new_patient_id = 'noise_' + str(np.random.randint(1e6))
#         patient_data['patient ID'] = new_patient_id

#         noise_df = pd.concat([noise_df, patient_data])

#     return noise_df
  
def add_random_jitter(patient_ids, data_path, use_absorbance_only, use_personalized_only, target_variable, augmentation_params):
    jitter_df = pd.DataFrame()
    data = filter_df(data_path, use_absorbance_only, use_personalized_only, target_variable)

    print('Jitter Level: ', augmentation_params['jitter_level'])

    # For each patient ID, create jitter data and add it to jitter_df
    for patient_id in patient_ids:
        patient_data = data[data['patient ID'] == patient_id].copy()

        # Only apply random jitter to selected columns
        excluded_columns = [target_variable, 'patient ID', 'collection time']
        jitter_columns = [col for col in data.columns if col not in excluded_columns]

        # Add random jitter to each value in the selected columns
        for column in jitter_columns:
            jitter = np.random.uniform(-augmentation_params['jitter_level'], augmentation_params['jitter_level'], size=patient_data[column].shape)
            patient_data[column] += jitter

        # Generate a new patient ID for the jitter data
        new_patient_id = 'jitter_' + str(np.random.randint(1e6))  # This generates a random integer between 0 and 1e6
        patient_data['patient ID'] = new_patient_id

        jitter_df = pd.concat([jitter_df, patient_data])

    return jitter_df


# def add_random_jitter(patient_ids, data_path, use_absorbance_only, use_personalized_only, target_variable, augmentation_params):
#     jitter_df = pd.DataFrame()
#     data = filter_df(data_path, use_absorbance_only, use_personalized_only, target_variable)

#     print('Jitter Level: ', augmentation_params['jitter_level'])

#     # Filter only continuous columns
#     continuous_data = filter_continuous_columns(data)

#     # For each patient ID, create jitter data and add it to jitter_df
#     for patient_id in patient_ids:
#         patient_data = continuous_data[continuous_data['patient ID'] == patient_id].copy()

#         # Only apply random jitter to selected columns
#         excluded_columns = [target_variable, 'patient ID', 'collection time']
#         jitter_columns = [col for col in patient_data.columns if col not in excluded_columns]

#         # Add random jitter to each value in the selected columns
#         for column in jitter_columns:
#             jitter = np.random.uniform(-augmentation_params['jitter_level'], augmentation_params['jitter_level'], size=patient_data[column].shape)
#             patient_data[column] += jitter

#         # Generate a new patient ID for the jitter data
#         new_patient_id = 'jitter_' + str(np.random.randint(1e6))
#         patient_data['patient ID'] = new_patient_id

#         jitter_df = pd.concat([jitter_df, patient_data])

#     return jitter_df

