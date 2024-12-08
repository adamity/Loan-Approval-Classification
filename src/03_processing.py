from utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def inspect(data):
    inspect_basic_structure(data)
    examine_summary_statistics(data)
    identify_missing_values(data)
    check_for_duplicates(data)
    check_for_constant_columns(data)
    check_data_types(data)


def convert_to_category(data, categorical_columns):
    data = data.copy()

    for column in categorical_columns:
        data[column] = data[column].astype('category')

    return data


def scale_columns(data, columns_to_scale, scaler_type='minmax'):
    data = data.copy()

    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid scaler type. Use 'minmax' or 'standard'.")

    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    return data


def label_encode_columns(data, binary_columns):
    data = data.copy()

    for column in binary_columns:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column])

    return data


def one_hot_encode_columns(data, nominal_columns):
    data = data.copy()
    return pd.get_dummies(data, columns=nominal_columns, drop_first=False)


def main():
    data = load_data('data/LoanApproval_Cleaned.csv')

    print("\033[34mBefore Data Processing\033[0m\n")
    inspect(data)

    categorical_columns = [
        'gender', 'education', 'home_ownership',
        'loan_intent', 'previous_defaults', 'loan_status'
    ]
    data = convert_to_category(data, categorical_columns)

    columns_to_scale = [
        'age', 'annual_income', 'employment_experience_years',
        'loan_amount', 'loan_interest_rate', 'loan_percent_income',
        'credit_history_length_years', 'credit_score'
    ]
    data = scale_columns(data, columns_to_scale, scaler_type='minmax')

    binary_columns = ['gender', 'previous_defaults', 'loan_status']
    data = label_encode_columns(data, binary_columns)

    nominal_columns = ['education', 'home_ownership', 'loan_intent']
    data = one_hot_encode_columns(data, nominal_columns)

    print("\033[34mAfter Data Processing\033[0m\n")
    inspect(data)

    save_data(data, 'data/LoanApproval_Preprocessed.csv')


if __name__ == "__main__":
    main()
