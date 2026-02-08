import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from pathlib import Path
import sys
import joblib
import warnings

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent.parent))
import config
from ml.preprocessing.feature_pipeline import ChurnPreprocessingPipeline


class CourseModels:
    """
    Course-required model implementation with hyperparameter tuning
    Demonstrates correct application of ML techniques for academic evaluation
    """

    def __init__(self, preprocessing_data=None):
        """Initialize models with preprocessing results"""
        print("\n" + "=" * 80)
        print("🎓 INITIALIZING COURSE-REQUIRED MODEL TRAINING")
        print("=" * 80)

        if preprocessing_data is None:
            # Run preprocessing pipeline
            print("📊 Running preprocessing pipeline...")
            preprocessor = ChurnPreprocessingPipeline()
            preprocessing_data = preprocessor.run_complete_preprocessing()

        # Extract preprocessing components
        self.X_train = preprocessing_data["X_train"]
        self.X_test = preprocessing_data["X_test"]
        self.y_train = preprocessing_data["y_train"]
        self.y_test = preprocessing_data["y_test"]
        self.feature_names = preprocessing_data["feature_names"]
        self.target_labels = preprocessing_data["target_labels"]
        self.preprocessing_pipeline = preprocessing_data["preprocessing_pipeline"]
        self.label_encoder = preprocessing_data["label_encoder"]

        print(f"✅ Preprocessing data loaded:")
        print(f"   Training set shape: {self.X_train.shape}")
        print(f"   Test set shape: {self.X_test.shape}")

        # Model storage
        self.logistic_regression_model = None
        self.logistic_regression_best_params = None
        self.logistic_regression_metrics = None

        self.decision_tree_model = None
        self.decision_tree_best_params = None
        self.decision_tree_metrics = None

        # Model comparison
        self.best_model = None
        self.best_model_name = None

        # Ensure directories exist
        config.create_directories()

    # ==================== LOGISTIC REGRESSION SECTION ====================

    def train_enhanced_logistic_regression(self):
        """
        Train enhanced LogisticRegression with hyperparameter tuning
        COURSE REQUIRED: Demonstrate correct application of classification algorithm
        Hyperparam tuning: Regularization (C parameter, L1 and L2)
        """
        print("\n" + "=" * 60)
        print("🔍 ENHANCED LOGISTIC REGRESSION TRAINING")
        print("=" * 60)

        # Define hyperparameter grid for tuning
        param_grid = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l2", "l1"],  # L1 and L2 regularization
            "solver": ["saga"],  # saga supports both L1 and L2
        }

        print(f"\n  🔧 HYPERPARAMETER TUNING GRID:")
        print(f"    C values: {param_grid['C']}")
        print(f"    Penalties: {param_grid['penalty']}")
        print(f"    Solver: {param_grid['solver']}")

        # Base model
        base_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )

        # Grid search with cross-validation
        print(f"\n  🔄 Performing GridSearchCV with 5-fold cross-validation...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(self.X_train, self.y_train)

        print(f"\n  ✅ GridSearchCV completed")
        self.logistic_regression_best_params = grid_search.best_params_
        self.logistic_regression_model = grid_search.best_estimator_

        print(f"\n  🏆 BEST HYPERPARAMETERS:")
        print(f"    C: {self.logistic_regression_best_params['C']}")
        print(f"    Penalty: {self.logistic_regression_best_params['penalty']}")
        print(f"    Solver: {self.logistic_regression_best_params['solver']}")
        print(f"    Best CV F1-Score: {grid_search.best_score_:.4f}")

        return self.logistic_regression_model

    def evaluate_logistic_regression(self):
        """Evaluate tuned LogisticRegression model"""
        print("\n" + "=" * 60)
        print("📈 EVALUATING LOGISTIC REGRESSION")
        print("=" * 60)

        # Make predictions
        y_pred = self.logistic_regression_model.predict(self.X_test)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        self.logistic_regression_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "y_pred": y_pred,
        }

        print(f"\n  📊 LOGISTIC REGRESSION PERFORMANCE:")
        print(f"    Accuracy:  {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1-Score:  {f1:.4f}")

        return self.logistic_regression_metrics

    def extract_logistic_regression_importance(self):
        """Extract feature importance from LogisticRegression coefficients"""
        print("\n" + "=" * 60)
        print("📊 LOGISTIC REGRESSION FEATURE IMPORTANCE")
        print("=" * 60)

        coefficients = self.logistic_regression_model.coef_[0]

        # Handle feature names length mismatch
        if len(self.feature_names) != len(coefficients):
            feature_names = [f"Feature_{i}" for i in range(len(coefficients))]
        else:
            feature_names = self.feature_names

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(coefficients),
            }
        ).sort_values("abs_coefficient", ascending=False)

        print(f"\n  🔝 TOP 10 MOST IMPORTANT FEATURES:")
        for idx, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            direction = "👆 increases" if row["coefficient"] > 0 else "👇 decreases"
            print(
                f"    {idx:2d}. {row['feature']:<30} {direction:15} churn (coef: {row['coefficient']:>7.4f})"
            )

        self.logistic_regression_importance = importance_df
        return importance_df

    # ==================== DECISION TREE SECTION ====================

    def train_decision_tree(self):
        """
        Train DecisionTreeClassifier with hyperparameter tuning
        COURSE REQUIRED: Demonstrate correct application of tree-based algorithm
        Hyperparameter tuning: max_depth, min_samples_split, criterion
        """
        print("\n" + "=" * 60)
        print("🌳 DECISION TREE TRAINING")
        print("=" * 60)

        # Define hyperparameter grid for tuning
        param_grid = {
            "max_depth": [3, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        }

        print(f"\n  🔧 HYPERPARAMETER TUNING GRID:")
        print(f"    max_depth: {param_grid['max_depth']}")
        print(f"    min_samples_split: {param_grid['min_samples_split']}")
        print(f"    criterion: {param_grid['criterion']}")

        # Base model
        base_model = DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        )

        # Grid search with cross-validation
        print(f"\n  🔄 Performing GridSearchCV with 5-fold cross-validation...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(self.X_train, self.y_train)

        print(f"\n  ✅ GridSearchCV completed")
        self.decision_tree_best_params = grid_search.best_params_
        self.decision_tree_model = grid_search.best_estimator_

        print(f"\n  🏆 BEST HYPERPARAMETERS:")
        print(f"    max_depth: {self.decision_tree_best_params['max_depth']}")
        print(
            f"    min_samples_split: {self.decision_tree_best_params['min_samples_split']}"
        )
        print(f"    criterion: {self.decision_tree_best_params['criterion']}")
        print(f"    Best CV F1-Score: {grid_search.best_score_:.4f}")

        return self.decision_tree_model

    def evaluate_decision_tree(self):
        """Evaluate tuned DecisionTreeClassifier model"""
        print("\n" + "=" * 60)
        print("📈 EVALUATING DECISION TREE")
        print("=" * 60)

        # Make predictions
        y_pred = self.decision_tree_model.predict(self.X_test)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        self.decision_tree_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "y_pred": y_pred,
        }

        print(f"\n  📊 DECISION TREE PERFORMANCE:")
        print(f"    Accuracy:  {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1-Score:  {f1:.4f}")

        return self.decision_tree_metrics

    def extract_decision_tree_importance(self):
        """Extract feature importance from DecisionTree"""
        print("\n" + "=" * 60)
        print("📊 DECISION TREE FEATURE IMPORTANCE")
        print("=" * 60)

        importances = self.decision_tree_model.feature_importances_

        # Handle feature names length mismatch
        if len(self.feature_names) != len(importances):
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        else:
            feature_names = self.feature_names

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

        print(f"\n  🔝 TOP 10 MOST IMPORTANT FEATURES:")
        for idx, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(
                f"    {idx:2d}. {row['feature']:<30} importance: {row['importance']:>7.4f}"
            )

        self.decision_tree_importance = importance_df
        return importance_df

    # ==================== MODEL COMPARISON ====================

    def compare_models(self):
        """Compare Logistic Regression and Decision Tree performance"""
        print("\n" + "=" * 60)
        print("⚖️  MODEL COMPARISON")
        print("=" * 60)

        # Create comparison dataframe
        comparison_df = pd.DataFrame(
            {
                "Model": ["Logistic Regression", "Decision Tree"],
                "Accuracy": [
                    self.logistic_regression_metrics["accuracy"],
                    self.decision_tree_metrics["accuracy"],
                ],
                "Precision": [
                    self.logistic_regression_metrics["precision"],
                    self.decision_tree_metrics["precision"],
                ],
                "Recall": [
                    self.logistic_regression_metrics["recall"],
                    self.decision_tree_metrics["recall"],
                ],
                "F1-Score": [
                    self.logistic_regression_metrics["f1_score"],
                    self.decision_tree_metrics["f1_score"],
                ],
            }
        )

        print("\n  📊 PERFORMANCE COMPARISON:")
        print(comparison_df.to_string(index=False))

        # Determine best model
        if (
            self.logistic_regression_metrics["f1_score"]
            > self.decision_tree_metrics["f1_score"]
        ):
            self.best_model = self.logistic_regression_model
            self.best_model_name = "Logistic Regression"
            best_metrics = self.logistic_regression_metrics
        else:
            self.best_model = self.decision_tree_model
            self.best_model_name = "Decision Tree"
            best_metrics = self.decision_tree_metrics

        print(f"\n  🏆 BEST MODEL: {self.best_model_name}")
        print(f"    F1-Score: {best_metrics['f1_score']:.4f}")
        print(f"    Accuracy: {best_metrics['accuracy']:.4f}")

        return comparison_df

    def save_all_models(self):
        """Save all trained models for deployment"""
        print("\n" + "=" * 60)
        print("💾 SAVING ALL MODELS")
        print("=" * 60)

        # Save Logistic Regression
        lr_path = config.MODELS_DIR / "logistic_regression_tuned.pkl"
        lr_artifacts = {
            "model": self.logistic_regression_model,
            "metrics": self.logistic_regression_metrics,
            "best_params": self.logistic_regression_best_params,
            "feature_names": self.feature_names,
            "target_labels": self.target_labels,
            "importance_df": self.logistic_regression_importance,
        }
        joblib.dump(lr_artifacts, lr_path)
        print(f"  ✅ Logistic Regression saved to: {lr_path}")

        # Save Decision Tree
        dt_path = config.MODELS_DIR / "decision_tree_tuned.pkl"
        dt_artifacts = {
            "model": self.decision_tree_model,
            "metrics": self.decision_tree_metrics,
            "best_params": self.decision_tree_best_params,
            "feature_names": self.feature_names,
            "target_labels": self.target_labels,
            "importance_df": self.decision_tree_importance,
        }
        joblib.dump(dt_artifacts, dt_path)
        print(f"  ✅ Decision Tree saved to: {dt_path}")

        # Save best model
        best_model_path = (
            config.MODELS_DIR
            / f"best_model_{self.best_model_name.lower().replace(' ', '_')}.pkl"
        )
        best_artifacts = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "metrics": (
                self.logistic_regression_metrics
                if self.best_model_name == "Logistic Regression"
                else self.decision_tree_metrics
            ),
            "feature_names": self.feature_names,
            "target_labels": self.target_labels,
        }
        joblib.dump(best_artifacts, best_model_path)
        print(f"  ✅ Best model ({self.best_model_name}) saved to: {best_model_path}")

        return lr_path, dt_path, best_model_path

    def run_course_model_training(self):
        """Execute complete course model training pipeline"""
        print("\n" + "=" * 80)
        print("🎓 RUNNING COMPLETE COURSE MODEL TRAINING")
        print("=" * 80)

        # Logistic Regression
        print("\n" + "━" * 80)
        print("PHASE 1: LOGISTIC REGRESSION")
        print("━" * 80)
        self.train_enhanced_logistic_regression()
        self.evaluate_logistic_regression()
        self.extract_logistic_regression_importance()

        # Decision Tree
        print("\n" + "━" * 80)
        print("PHASE 2: DECISION TREE")
        print("━" * 80)
        self.train_decision_tree()
        self.evaluate_decision_tree()
        self.extract_decision_tree_importance()

        # Comparison and selection
        print("\n" + "━" * 80)
        print("PHASE 3: MODEL COMPARISON")
        print("━" * 80)
        comparison_df = self.compare_models()

        # Save models
        print("\n" + "━" * 80)
        print("PHASE 4: MODEL PERSISTENCE")
        print("━" * 80)
        self.save_all_models()

        print("\n" + "=" * 80)
        print("✅ COURSE MODEL TRAINING COMPLETE")
        print("=" * 80)

        return {
            "logistic_regression": self.logistic_regression_metrics,
            "decision_tree": self.decision_tree_metrics,
            "best_model": self.best_model_name,
            "comparison": comparison_df,
        }


if __name__ == "__main__":
    # Run course model training
    course_models = CourseModels()
    results = course_models.run_course_model_training()
