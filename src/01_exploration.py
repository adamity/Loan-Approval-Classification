from utils import *


def examine_categorical_columns(data):
    print("\033[33mData Value Counts (Person Gender)\033[0m")
    print(data['person_gender'].value_counts(), "\n")

    print("\033[33mData Value Counts (Person Education)\033[0m")
    print(data['person_education'].value_counts(), "\n")

    print("\033[33mData Value Counts (Person Home Ownership)\033[0m")
    print(data['person_home_ownership'].value_counts(), "\n")

    print("\033[33mData Value Counts (Loan Intent)\033[0m")
    print(data['loan_intent'].value_counts(), "\n")

    print("\033[33mData Value Counts (Previous Loan Defaults on File)\033[0m")
    print(data['previous_loan_defaults_on_file'].value_counts(), "\n")

    print("\033[33mData Value Counts (Loan Status)\033[0m")
    print(data['loan_status'].value_counts(), "\n")


def main():
    data = load_data('data/LoanApproval_Raw.csv')

    inspect_basic_structure(data)
    examine_summary_statistics(data)
    examine_categorical_columns(data)
    identify_missing_values(data)
    check_for_duplicates(data)
    check_for_constant_columns(data)
    check_data_types(data)
    detect_outliers(data)


if __name__ == "__main__":
    main()
