from data_management import load_datasets
from pipeline import sales_pipe
"""train pipeline on data"""

def run_training()-> None:
    sales = load_datasets()

    """divide target and features"""
    X_train, y_train = trainset[:, 1:], trainset[:, 0:1]

    """fit pipeline"""
    sales_pipe.fit(X_train, y_train)



if __name__ == '__main__':
    run_training()

