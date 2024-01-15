import pathlib #for working with file paths
import joblib #for loading/saving models
import sys #for working with command line arguments
import yaml #for working with yaml files
import pandas as pd #for working with dataframe
from sklearn import metrics #for evaluation metrics
from sklearn import tree #for tree based model
from dvclive import Live #for logging metrics and plots during training
from matplotlib import pyplot as plt #for creating plots


def evaluate(model, X, y, split, live, save_path):
    """
    Dump all evaluation metrics and plots for given datasets.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        X (pandas.DataFrame): Input DF.
        y (pamdas.Series): Target column.
        split (str): Dataset name.
        live (dvclive.Live): Dvclive instance.
        save_path (str): Path to save the metrics.
    """

    predictions_by_class = model.predict_proba(X) #returns probabilities for both the classes.
    predictions = predictions_by_class[:, 1] #storing probabilities of positive class

    # Use dvclive to log a few simple metrics...
    avg_prec = metrics.average_precision_score(y, predictions)
    roc_auc = metrics.roc_auc_score(y, predictions)
    if not live.summary:
        live.summary = {"avg_prec": {}, "roc_auc": {}} #making dictionary of metrics
    live.summary["avg_prec"][split] = avg_prec #update the metrics
    live.summary["roc_auc"][split] = roc_auc #update the metrics
    # ... and plots...
    # ... like an roc plot...
    live.log_sklearn_plot("roc", y, predictions, name=f"roc/{split}")#log_sklearn_plot function is used to log plots specifically designed for scikit-learn metrics.
    # ... and precision recall plot...
    # ... which passes `drop_intermediate=True` to the sklearn method...
    live.log_sklearn_plot(
        "precision_recall",
        y,
        predictions,
        name=f"prc/{split}",
        drop_intermediate=True,
    )
    # ... and confusion matrix plot
    #It logs the confusion matrix for the given target labels y 
    #and the class predictions obtained by selecting the class with the highest probability from predictions_by_class
    live.log_sklearn_plot(
        "confusion_matrix",
        y,
        predictions_by_class.argmax(-1),
        name=f"cm/{split}",
    )


def save_importance_plot(live, model, feature_names):
    """
    Save feature importance plot.

    Args:
        live (dvclive.Live): DVCLive instance.
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        feature_names (list): List of feature names.
    """
    #fig: The whole plotting area or canvas.
    #axes: The individual plots or subplots within the Figure. Multiple subplots can exist within a single Figure.
    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    axes.set_ylabel("Mean decrease in impurity")

    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).nlargest(n=10)
    forest_importances.plot.bar(ax=axes)

    live.log_image("importance.png", fig) #log the generated image in logging directory of dvclive
    #This allows tracking and visualizing the feature importance plot during the training process.

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    # TODO - Optionally add visualization params as well
    # params_file = home_dir.as_posix() + '/params.yaml'
    # params = yaml.safe_load(open(params_file))["train_model"]

    model_file = sys.argv[1]
    # Load the model.
    model = joblib.load(model_file)
    
    # Load the data.
    input_file = sys.argv[2]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/dvclive'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    TARGET = 'Class'
    train_features = pd.read_csv(data_path + '/train.csv')
    X_train = train_features.drop(TARGET, axis=1)
    y_train = train_features[TARGET]
    feature_names = X_train.columns.to_list()

    test_features = pd.read_csv(data_path + '/test.csv')
    X_test = test_features.drop(TARGET, axis=1)
    y_test = test_features[TARGET]

    # Evaluate train and test datasets.
    with Live(output_path, dvcyaml=False) as live:
        evaluate(model, X_train, y_train, "train", live, output_path)
        evaluate(model, X_test, y_test, "test", live, output_path)

        # Dump feature importance plot.
        save_importance_plot(live, model, feature_names)

if __name__ == "__main__":
    main()
