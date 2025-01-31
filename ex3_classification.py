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
        self.X = None
        self.y = None
        self.results = {}  # Stores accuracy & AUC for each method
        self.best_method = None
        self.best_model = None
        self.best_X_train = None
        self.best_X_test = None
        self.best_y_train = None
        self.best_y_test = None
        self.best_y_pred = None
        self.best_y_prob = None

    def preprocess(self):
        """Mandatory Preprocessing: Drop NA, remove duplicates, handle outliers, encode categorical columns."""
        print("\nStarting preprocessing...")

        # Drop NA and Duplicates
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)

        # Handle outliers using quantile clipping
        for col in self.df.select_dtypes(include=[np.number]).columns:
            lower_bound = self.df[col].quantile(0.01)
            upper_bound = self.df[col].quantile(0.99)
            self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)

        # Encode categorical variables
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_cols:
            self.df[col] = le.fit_transform(self.df[col].astype(str))

        # Split dataset into features and target
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]

        # If target column is categorical, encode it
        if self.y.dtype == 'object':
            self.y = le.fit_transform(self.y.astype(str))

        print("Preprocessing complete.\n")

    def train_model(self, model, method_name, X_train, X_test, y_train, y_test):
        """Train the model and store the performance metrics."""
        print(f"\nTraining model: {method_name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None

        # Compute Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
        self.results[method_name] = {"accuracy": accuracy, "auc": auc_score}

        # Store the best model
        if self.best_method is None or (accuracy + auc_score) > (self.results[self.best_method]["accuracy"] + self.results[self.best_method]["auc"]):
            self.best_method = method_name
            self.best_model = model
            self.best_X_train, self.best_X_test, self.best_y_train, self.best_y_test = X_train, X_test, y_train, y_test
            self.best_y_pred, self.best_y_prob = y_pred, y_prob

    def apply_resampling(self, resampling_method, method_name):
        """Apply SMOTE, SMOTETomek, or SMOTEENN."""
        print(f"\nApplying {method_name} ...")
        sampler = resampling_method()
        X_resampled, y_resampled = sampler.fit_resample(self.X, self.y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42)
        self.train_model(DecisionTreeClassifier(class_weight="balanced", random_state=42), method_name, X_train, X_test, y_train, y_test)

    def use_important_features(self):
        """Train using only important features (threshold = 0.01)."""
        print("\nTraining with only important features...")
        importance_threshold = 0.01
        feature_importances = pd.Series(self.best_model.feature_importances_, index=self.X.columns)
        selected_features = feature_importances[feature_importances > importance_threshold].index
        X_selected = self.X[selected_features]

        X_train, X_test, y_train, y_test = train_test_split(X_selected, self.y, test_size=0.3, stratify=self.y, random_state=42)
        self.train_model(DecisionTreeClassifier(class_weight="balanced", random_state=42), "Decision Tree (Important Features)", X_train, X_test, y_train, y_test)

    def tune_hyperparameters_with_grid_search(self, method_name):
        """Apply GridSearchCV or RandomizedSearchCV."""
        print(f"\nApplying {method_name} ...")
        param_grid = {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 3, 5],
            'criterion': ['gini', 'entropy']
        }
        search = GridSearchCV(DecisionTreeClassifier(class_weight="balanced", random_state=42), param_grid, cv=5)
        search.fit(self.X, self.y)
        best_model = search.best_estimator_

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, stratify=self.y, random_state=42)
        self.train_model(best_model, method_name, X_train, X_test, y_train, y_test)
    
    def tune_hyperparameters_with_randomized_search(self, method_name):
        """Apply GridSearchCV or RandomizedSearchCV."""
        print(f"\nApplying {method_name} ...")

        param_distributions = {
            'max_depth': randint(3, 30),
            'min_samples_split': randint(2, 15),
            'min_samples_leaf': randint(1, 10),
            'criterion': ['gini', 'entropy'],
            'max_features': [None, 'sqrt', 'log2']
        }
        search = RandomizedSearchCV(DecisionTreeClassifier(class_weight="balanced", random_state=42), param_distributions, cv=5)
        search.fit(self.X, self.y)
        best_model = search.best_estimator_

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, stratify=self.y, random_state=42)
        self.train_model(best_model, method_name, X_train, X_test, y_train, y_test)

    def train_random_forest(self, method_name):
        """Train a Random Forest classifier."""
        print(f"\nTraining {method_name} ...")
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, stratify=self.y, random_state=42)
        self.train_model(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42), method_name, X_train, X_test, y_train, y_test)

    def run_pipeline(self):
        """Execute the full classification pipeline."""
        self.preprocess()

        # Train Decision Tree (Baseline)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, stratify=self.y, random_state=42)
        self.train_model(DecisionTreeClassifier(class_weight="balanced", random_state=42), "Decision Tree (Baseline)", X_train, X_test, y_train, y_test)

        # Apply resampling methods
        self.apply_resampling(SMOTE, "Decision Tree (SMOTE)")
        self.apply_resampling(SMOTETomek, "Decision Tree (SMOTETomek)")
        self.apply_resampling(SMOTEENN, "Decision Tree (SMOTEENN)")

        # Train with important features
        self.use_important_features()

        # Apply hyperparameter tuning
        self.tune_hyperparameters_with_grid_search("Decision Tree (GridSearchCV)")
        self.tune_hyperparameters_with_randomized_search("Decision Tree (RandomizedSearchCV)")
        #self.tune_hyperparameters(GridSearchCV, "Decision Tree (GridSearchCV)")
        #self.tune_hyperparameters(RandomizedSearchCV, "Decision Tree (RandomizedSearchCV)")

        # Train Random Forest variations
        self.train_random_forest("Random Forest (Baseline)")
        self.train_random_forest("Random Forest (Important Features)")

        # Display best result
        print(f"\nBest model: {self.best_method}")
        print(classification_report(self.best_y_test, self.best_y_pred))

        # If best model is a Decision Tree, plot visuals
        if "Decision Tree" in self.best_method:
            self.plot_results()

    def plot_results(self):
        """Plot confusion matrix, ROC curve, and Decision Tree visualization."""
        cm = confusion_matrix(self.best_y_test, self.best_y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.show()

        fpr, tpr, _ = roc_curve(self.best_y_test, self.best_y_prob)
        roc_auc = auc(fpr, tpr)
        """ plt.plot(fpr, tpr)
        plt.show() """

        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")  # Diagonal Line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.show()
    

        plt.figure(figsize=(15, 10))
        plot_tree(self.best_model, feature_names=self.best_X_train.columns, filled=True)
        plt.show()

# Run the program
if __name__ == "__main__":
    pipeline = ClassificationPipeline()
    pipeline.run_pipeline()
