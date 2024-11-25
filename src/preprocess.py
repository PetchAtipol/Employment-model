import pandas as pd

def preprocess():
    # Load the CSV file
    # file_path = '/content/อัตราการมีงานทำต่อประชากรวัยแรงงาน.csv'
    file_path = 'data/raw/อัตราการมีงานทำต่อประชากรวัยแรงงาน.csv'
    data = pd.read_csv(file_path)

    # # Filter the data to only include rows where region is 'ทั่วประเทศ' and area is 'รวม'
    # filtered_data = data[(data['region'] == 'ทั่วประเทศ') & (data['area'] == 'รวม')]

    # Delete rows where `level_of_edu` column has the value 'Unknown'
    # filtered_data = data[data['level_of_edu'] != 'การศึกษาอื่นๆ']

    # # Delete rows where `level_of_edu` column has the value 'Unknown'
    # filtered_data = filtered_data[filtered_data['level_of_edu'] != 'ไม่ทราบ']

    # Perform one-hot encoding on the 'region', 'area', and 'level_of_edu' columns
    binary_coded_data = pd.get_dummies(data, columns=['quarter','region', 'area', 'level_of_edu'])

    binary_coded_data = binary_coded_data.drop(columns=['unit','source'])
    binary_coded_data = binary_coded_data.replace({True: 1, False: 0})

    # Save the filtered data to a new CSV file
    binary_coded_data_path = 'data/processed/data_binary_clean.csv'
    binary_coded_data.to_csv(binary_coded_data_path, index=False)

    # print(filtered_data)

    # Display rows with missing values
    # rows_with_missing = filtered_data[filtered_data.isnull().any(axis=1)]
    # print(rows_with_missing)

    # Display duplicate rows
    # duplicates = data[data.duplicated()]
    # print(duplicates)