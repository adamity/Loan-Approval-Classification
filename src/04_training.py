import os
import joblib
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn import metrics


def evaluate_model(y_test, y_pred):
    print("\033[32mModel Evaluation\033[0m\n")

    print("\033[33mClassification Report\033[0m")
    print(metrics.classification_report(y_test, y_pred), "\n")


def save_model(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)


def main():
    data = load_data('data/LoanApproval_Preprocessed.csv')

    X = data.drop('loan_status', axis=1)
    Y = data['loan_status']

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0)

    # Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)

    print("\033[34mRandom Forest Classifier\033[0m\n")
    evaluate_model(y_test, y_pred)

    # SVM Classifier

    # Linear Kernel
    svc_linear = SVC(kernel='linear')
    svc_linear.fit(x_train, y_train)
    y_pred = svc_linear.predict(x_test)

    print("\033[34mSupport Vector Machine Classifier (Linear Kernel)\033[0m\n")
    evaluate_model(y_test, y_pred)

    # RBF Kernel
    svc_rbf = SVC(kernel='rbf')
    svc_rbf.fit(x_train, y_train)
    y_pred = svc_rbf.predict(x_test)

    print("\033[34mSupport Vector Machine Classifier (RBF Kernel)\033[0m\n")
    evaluate_model(y_test, y_pred)

    # Poly Kernel
    svc_poly = SVC(kernel='poly')
    svc_poly.fit(x_train, y_train)
    y_pred = svc_poly.predict(x_test)

    print("\033[34mSupport Vector Machine Classifier (Poly Kernel)\033[0m\n")
    evaluate_model(y_test, y_pred)

    # LIGHTGBM Classifier
    lgbm = LGBMClassifier()
    lgbm.fit(x_train, y_train)
    y_pred = lgbm.predict(x_test)

    print("\033[34mLightGBM Classifier\033[0m\n")
    evaluate_model(y_test, y_pred)

    # Save Models
    save_model(rfc, 'models/RandomForestClassifier.pkl')
    save_model(svc_linear, 'models/SVC_Linear.pkl')
    save_model(svc_rbf, 'models/SVC_RBF.pkl')
    save_model(svc_poly, 'models/SVC_Poly.pkl')
    save_model(lgbm, 'models/LightGBMClassifier.pkl')


if __name__ == "__main__":
    main()
