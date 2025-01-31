import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from scipy.stats import randint

class ClassificationPipeline:
    def __init__(self):
        """Initialize the pipeline."""
        self.dataset_path = input("Enter the path to your dataset: ")
        self.df = pd.read_csv(self.dataset_path)
        self.target_column = input("Enter the name of the target column: ")
        """ADD MORE LOGIC IF NECESSARY"""


    """ ADD AS MANY FUNCTIONS AS NECESSARY"""

    def run_pipeline(self):
        """ADD THE NECESSARY LOGIC TO COMPLETE THIS FUNCTION"""

        
        """DO NOT CHANGE THE FOLLOWING TWO LINES OF CODE. THESE ARE NEEDED TO TEST YOUR MODEL PERFORMANCE FOR THE EXERCISE."""
        print(f"Best Accuracy Score: {self.results[self.best_method]['accuracy']:.4f}")
        print(f"Best AUC Score: {self.results[self.best_method]['auc']:.4f}")



if __name__ == "__main__":
    pipeline = ClassificationPipeline()
    pipeline.run_pipeline()
