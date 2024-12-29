from utils import load_data
from pycaret.classification import *


def main():
    # Load raw data
    data = load_data('data/LoanApproval_Raw.csv')

    # Initial setup for the entire workflow
    setup(
        data=data,
        target='loan_status',
        session_id=123,
        verbose=True,
        profile=True,
        remove_multicollinearity=True,
        remove_outliers=True,
        normalize=True,
        transformation=True,
        categorical_features=[
            'person_gender', 'person_education', 'person_home_ownership',
            'loan_intent', 'previous_loan_defaults_on_file'
        ]
    )

    # Train models and save the best one
    best_model = compare_models()
    evaluate_model(best_model)
    save_model(best_model, 'models/Best_Model')


if __name__ == "__main__":
    main()
