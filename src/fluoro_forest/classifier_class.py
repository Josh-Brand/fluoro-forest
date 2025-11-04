from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, average_precision_score # Import average_precision_score
)
from sklearn.preprocessing import label_binarize # Import label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import pandas as pd
import numpy as np

# classifier (random forest) applied to annotated data after running the app, derived from shallow copies of data and handles expression / plot data for cell type prediction
class classify_cells:
    def __init__(self, core_class):
        self.expression_data = core_class.expression_data.copy()
        self.plot_data = core_class.plot_df.copy()
        annotated_cells = core_class.annotations
        self.expression_data['annotations'] = ''
        self.expression_data.loc[annotated_cells.keys(), 'annotations'] = list(annotated_cells.values())
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.max_smote_neighbors = 5 # Define max neighbors

    # set splitting parameters (this or crossfold validation) this has more control over splitting size
    def custom_train_test_split(self, X, y, test_size = 0.2, random_state = None): # default test is 20% of data
        classes = np.unique(y)
        train_indices = []
        test_indices = []

        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            n_samples = len(cls_indices)
            n_test = max(1, int(test_size * n_samples))
            n_train = n_samples - n_test

            # state control if needed to reproduce results
            np.random.seed(random_state)
            np.random.shuffle(cls_indices)

            train_indices.extend(cls_indices[:n_train])
            test_indices.extend(cls_indices[n_train:])

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        return X_train, X_test, y_train, y_test

        # train data
    def train(self, use_params, n_trees = 200, split = 0.2, random_state = 5,
              use_imbalanced_rf = False, use_smote = False):

        # filter to uuse parameters and only annotated cells

        use_data = self.expression_data[self.expression_data['annotations'] != '']

        filtered_params = [param for param in use_params if param in self.expression_data.columns]

        X = use_data[filtered_params]
        y = use_data['annotations']

        # class counts used, important for smote (minority class over sample)
        class_counts = y.value_counts()
        #print("\nInitial Class Counts:")
        #print(class_counts)

        smote_k_neighbors = {}
        for cls in class_counts.index:
            smote_k_neighbors[cls] = min(class_counts[cls] - 1, self.max_smote_neighbors)

            if smote_k_neighbors[cls] < 1: # needs at least 2 cells for knn in smote
                print(f"  Warning: Class {cls} has too few samples. SMOTE will not be applied.")

        # filter to cminimum class size (only for smote)
        min_samples = 2
        valid_classes = class_counts[class_counts >= min_samples].index

        # Filter data to keep only valid classes
        X_valid = X[y.isin(valid_classes)]
        y_valid = y[y.isin(valid_classes)]

        print("\nClass Counts after filtering classes with < 2 samples:") # smote speciifc
        print(y_valid.value_counts())

        # splitting data
        if split:
            X_filtered = X_valid
            y_filtered = y_valid

            try:
                self.X_train, self.X_test, self.y_train, self.y_test = self.custom_train_test_split(
                    X_filtered, y_filtered, test_size=split, random_state=random_state)
                self.plot_data.loc[:, 'used_in_training'] = False
                self.plot_data.loc[self.X_train.index, 'used_in_training'] = True # prevents data leak in plots

            except ValueError as e:
                print(f"Error during custom train_test_split: {e}")
                return
        else:
            self.X_train, self.y_train = X, y
            self.X_test, self.y_test = None, None

        print("\nClass counts in y_train after train_test_split:")
        print(pd.Series(self.y_train).value_counts())
        print("\nClass counts in y_test after train_test_split:")
        print(pd.Series(self.y_test).value_counts())

        # implementing smote
        if use_smote:
            #  minimum class count across all classes in the training set
            min_class_count = min(self.y_train.value_counts())
            print('min class count', min_class_count)
            k_neighbors = min(min_class_count - 1, self.max_smote_neighbors)
            print(f"\napplying smote to full *training* set with knn={k_neighbors}")

            try:
                smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)  # Resample training data
                print("\nClass counts in y_train after smote:")
                print(pd.Series(self.y_train).value_counts())
            except ValueError as e:
                print(f"ValueError : {e}. smote cant be applied.")

        # initialize and train ramdom forest (class balancing can be done manually with sampling or here)
        if use_imbalanced_rf:
            self.model = BalancedRandomForestClassifier(n_estimators = n_trees, random_state = random_state)
        else:
            self.model = RandomForestClassifier(n_estimators = n_trees, random_state = random_state)

        self.model.fit(self.X_train, self.y_train)

        # If test set evaluates the model
        if self.X_test is not None:
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"model Accuracy: {accuracy:.2f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))

        # used to fit full model on data
    def fit(self, new_data = None):
        if self.model is None:
            raise ValueError("model not trined, use train() method first")

        if new_data is None:
            # use all unannotated data with trained model to predict cell types
            new_data = self.expression_data[self.model.feature_names_in_]

        predictions = self.model.predict(new_data)
        probabilities = self.model.predict_proba(new_data)

        # predictions and probabilities added to dataframe
        self.expression_data['predicted_annotation'] = predictions
        for i, class_name in enumerate(self.model.classes_):
            self.expression_data[f'prob_{class_name}'] = probabilities[:, i]


        # used for evaluating models
    def k_fold_cross_validation(self, use_params, n_splits = 5, n_trees = 200, random_state = 5,
                                 use_imbalanced_rf = False, use_smote = False):
        use_data = self.expression_data[self.expression_data['annotations'] != '']
        filtered_params = [param for param in use_params if param in self.expression_data.columns]

        X = use_data[filtered_params]
        y = use_data['annotations']

        # filter out classes with fewer samples than n_splits (rare cell types will not be predicted well)
        class_counts = y.value_counts()
        valid_classes = class_counts[class_counts >= n_splits].index
        X_valid = X[y.isin(valid_classes)]
        y_valid = y[y.isin(valid_classes)]

        if len(valid_classes) < 2:
            raise ValueError("not enough valid classes for cross-validation")

        skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)
        metrics_list = []

        # Get all unique classes from the full dataset for consistent binarization
        all_classes_in_data = np.unique(y_valid)

        for fold, (train_index, val_index) in enumerate(skf.split(X_valid, y_valid), 1):
            X_train, X_val = X_valid.iloc[train_index], X_valid.iloc[val_index]
            y_train, y_val = y_valid.iloc[train_index], y_valid.iloc[val_index]

            if use_smote:
                smote_k = min(self.max_smote_neighbors, min(y_train.value_counts()) - 1)
                if smote_k >= 1:
                    smote = SMOTE(random_state = random_state, k_neighbors = smote_k)
                    X_train, y_train = smote.fit_resample(X_train, y_train)

            # Initialize model
            if use_imbalanced_rf:
                model = BalancedRandomForestClassifier(n_estimators = n_trees, random_state = random_state)
            else:
                model = RandomForestClassifier(n_estimators = n_trees, random_state = random_state)

            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Get probability predictions for Average Precision
            y_proba = model.predict_proba(X_val)

            # Binarize y_val for average precision calculation
            # Use all_classes_in_data to ensure consistent columns in binarized output
            y_val_binarized = label_binarize(y_val, classes=all_classes_in_data)

            # Calculate overall metrics
            accuracy = accuracy_score(y_val, y_pred)
            macro_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
            weighted_f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

            # Calculate Average Precision (macro and weighted)
            # Ensure that the columns of y_proba match the binarized classes
            # You might need to reorder y_proba columns if model.classes_ is different from all_classes_in_data
            # For simplicity, assume model.classes_ will match the order of all_classes_in_data or can be handled
            # by label_binarize's internal class handling. If not, map them.
            # A safer approach for y_proba if model.classes_ is not consistent with all_classes_in_data:
            y_proba_aligned = np.zeros((y_proba.shape[0], len(all_classes_in_data)))
            for i, cls in enumerate(model.classes_):
                if cls in all_classes_in_data:
                    col_idx = np.where(all_classes_in_data == cls)[0][0]
                    y_proba_aligned[:, col_idx] = y_proba[:, i]

            macro_avg_precision = average_precision_score(y_val_binarized, y_proba_aligned, average='macro')
            weighted_avg_precision = average_precision_score(y_val_binarized, y_proba_aligned, average='weighted')

            # Store overall metrics for the fold
            metrics_list.append({
                'fold': fold,
                'metric_type': 'overall',
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1,
                'macro_avg_precision': macro_avg_precision,
                'weighted_avg_precision': weighted_avg_precision
            })

            # Per-class metrics
            for cls in model.classes_:
                f1 = f1_score(y_val, y_pred, labels=[cls], average='macro', zero_division=0)
                precision = precision_score(y_val, y_pred, labels=[cls], average='macro', zero_division=0)
                recall = recall_score(y_val, y_pred, labels=[cls], average='macro', zero_division=0)

                # Find the column index for the current class in y_proba
                try:
                    class_idx_in_model_classes = list(model.classes_).index(cls)
                    y_proba_cls = y_proba[:, class_idx_in_model_classes]

                    # Binarize y_val for this specific class for AP calculation
                    y_val_cls_binarized = (y_val == cls).astype(int)

                    # Average precision for individual class
                    # Check if there are positive samples for the class in the validation set
                    if np.sum(y_val_cls_binarized) > 0:
                        ap_cls = average_precision_score(y_val_cls_binarized, y_proba_cls)
                    else:
                        ap_cls = np.nan # No positive samples, AP is undefined or 0
                except ValueError:
                    ap_cls = np.nan # Class not present in model's learned classes for some reason


                metrics_list.append({
                    'fold': fold,
                    'metric_type': 'per_class',
                    'class': cls,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'avg_precision': ap_cls # Per-class average precision
                })

        metrics_df = pd.DataFrame(metrics_list)
        return metrics_df