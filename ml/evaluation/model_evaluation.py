"""
Model Evaluation Module (Section 5.1)
Purpose: Demonstrate appropriate evaluation metrics as required by course
Includes: Accuracy, Precision, Recall, F1-score, Confusion Matrices, ROC Curves
Academic evaluation for ML model performance assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    classification_report,
    precision_recall_curve,
)
from pathlib import Path
import sys
import joblib
import warnings

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


class ModelEvaluation:
    """
    Comprehensive model evaluation for academic assessment
    Demonstrates quality evaluation of ML models
    """

    def __init__(self):
        """Initialize evaluation module"""
        print("\n" + "=" * 80)
        print("📊 INITIALIZING MODEL EVALUATION MODULE")
        print("=" * 80)

        # Load trained models and preprocessing data
        self._load_models_and_data()
        self.evaluation_results = {}

        # Configure plots
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10

        # Ensure directories exist
        config.create_directories()

    def _load_models_and_data(self):
        """Load models and preprocessing components"""
        print("\n  🔄 Loading trained models and preprocessing data...")

        try:
            # Load baseline model
            baseline_path = config.MODELS_DIR / "baseline_logistic_regression.pkl"
            baseline_artifacts = joblib.load(baseline_path)
            self.baseline_model = baseline_artifacts["model"]
            self.baseline_metrics_saved = baseline_artifacts["metrics"]
            self.baseline_importance = baseline_artifacts["importance_df"]
            print(f"  ✅ Baseline model loaded")
        except Exception as e:
            print(f"  ⚠️  Could not load baseline model: {e}")
            self.baseline_model = None

        try:
            # Load Logistic Regression
            lr_path = config.MODELS_DIR / "logistic_regression_tuned.pkl"
            lr_artifacts = joblib.load(lr_path)
            self.lr_model = lr_artifacts["model"]
            self.lr_metrics_saved = lr_artifacts["metrics"]
            self.lr_importance = lr_artifacts["importance_df"]
            self.lr_best_params = lr_artifacts["best_params"]
            print(f"  ✅ Logistic Regression model loaded")
        except Exception as e:
            print(f"  ⚠️  Could not load LR model: {e}")
            self.lr_model = None

        try:
            # Load Decision Tree
            dt_path = config.MODELS_DIR / "decision_tree_tuned.pkl"
            dt_artifacts = joblib.load(dt_path)
            self.dt_model = dt_artifacts["model"]
            self.dt_metrics_saved = dt_artifacts["metrics"]
            self.dt_importance = dt_artifacts["importance_df"]
            self.dt_best_params = dt_artifacts["best_params"]
            print(f"  ✅ Decision Tree model loaded")
        except Exception as e:
            print(f"  ⚠️  Could not load DT model: {e}")
            self.dt_model = None

        try:
            # Load preprocessing pipeline
            pipeline_path = config.PREPROCESSING_DIR / "preprocessing_pipeline.pkl"
            pipeline_components = joblib.load(pipeline_path)
            self.preprocessing_pipeline = pipeline_components["preprocessing_pipeline"]
            self.label_encoder = pipeline_components["label_encoder"]
            self.feature_names = pipeline_components["feature_names"]
            self.target_labels = pipeline_components["target_labels"]
            print(f"  ✅ Preprocessing pipeline loaded")
        except Exception as e:
            print(f"  ⚠️  Could not load preprocessing pipeline: {e}")

        # Load test data
        try:
            from ml.preprocessing.feature_pipeline import ChurnPreprocessingPipeline

            preprocessor = ChurnPreprocessingPipeline()
            prep_data = preprocessor.run_complete_preprocessing()
            self.X_test = prep_data["X_test"]
            self.y_test = prep_data["y_test"]
            print(f"  ✅ Test dataset loaded: {self.X_test.shape}")
        except Exception as e:
            print(f"  ⚠️  Could not load test data: {e}")

    # ==================== EVALUATION METRICS SECTION ====================

    def calculate_all_metrics(self):
        """Calculate all course-required metrics for both models"""
        print("\n" + "=" * 60)
        print("📈 CALCULATING EVALUATION METRICS (COURSE REQUIRED)")
        print("=" * 60)

        if self.lr_model is None or self.dt_model is None:
            print("  ⚠️  Models not loaded, cannot calculate metrics")
            return

        # Logistic Regression metrics
        print("\n  🔹 LOGISTIC REGRESSION METRICS:")
        lr_y_pred = self.lr_model.predict(self.X_test)
        lr_y_pred_proba = self.lr_model.predict_proba(self.X_test)[:, 1]

        lr_metrics = {
            "accuracy": accuracy_score(self.y_test, lr_y_pred),
            "precision": precision_score(self.y_test, lr_y_pred),
            "recall": recall_score(self.y_test, lr_y_pred),
            "f1_score": f1_score(self.y_test, lr_y_pred),
            "roc_auc": roc_auc_score(self.y_test, lr_y_pred_proba),
            "y_pred": lr_y_pred,
            "y_pred_proba": lr_y_pred_proba,
            "confusion_matrix": confusion_matrix(self.y_test, lr_y_pred),
        }

        print(f"    Accuracy:  {lr_metrics['accuracy']:.4f}")
        print(f"    Precision: {lr_metrics['precision']:.4f}")
        print(f"    Recall:    {lr_metrics['recall']:.4f}")
        print(f"    F1-Score:  {lr_metrics['f1_score']:.4f}")
        print(f"    ROC-AUC:   {lr_metrics['roc_auc']:.4f}")

        # Decision Tree metrics
        print("\n  🔹 DECISION TREE METRICS:")
        dt_y_pred = self.dt_model.predict(self.X_test)
        dt_y_pred_proba = self.dt_model.predict_proba(self.X_test)[:, 1]

        dt_metrics = {
            "accuracy": accuracy_score(self.y_test, dt_y_pred),
            "precision": precision_score(self.y_test, dt_y_pred),
            "recall": recall_score(self.y_test, dt_y_pred),
            "f1_score": f1_score(self.y_test, dt_y_pred),
            "roc_auc": roc_auc_score(self.y_test, dt_y_pred_proba),
            "y_pred": dt_y_pred,
            "y_pred_proba": dt_y_pred_proba,
            "confusion_matrix": confusion_matrix(self.y_test, dt_y_pred),
        }

        print(f"    Accuracy:  {dt_metrics['accuracy']:.4f}")
        print(f"    Precision: {dt_metrics['precision']:.4f}")
        print(f"    Recall:    {dt_metrics['recall']:.4f}")
        print(f"    F1-Score:  {dt_metrics['f1_score']:.4f}")
        print(f"    ROC-AUC:   {dt_metrics['roc_auc']:.4f}")

        self.lr_metrics = lr_metrics
        self.dt_metrics = dt_metrics
        self.evaluation_results["metrics"] = {
            "logistic_regression": lr_metrics,
            "decision_tree": dt_metrics,
        }

        return lr_metrics, dt_metrics

    def generate_classification_reports(self):
        """Generate detailed classification reports"""
        print("\n" + "=" * 60)
        print("📋 CLASSIFICATION REPORTS")
        print("=" * 60)

        # Logistic Regression report
        print("\n  🔹 LOGISTIC REGRESSION CLASSIFICATION REPORT:")
        lr_report = classification_report(
            self.y_test, self.lr_metrics["y_pred"], target_names=self.target_labels
        )
        print(lr_report)

        # Decision Tree report
        print("\n  🔹 DECISION TREE CLASSIFICATION REPORT:")
        dt_report = classification_report(
            self.y_test, self.dt_metrics["y_pred"], target_names=self.target_labels
        )
        print(dt_report)

        self.lr_report = lr_report
        self.dt_report = dt_report

    def compare_and_select_best_model(self):
        """Compare models and select the best one"""
        print("\n" + "=" * 60)
        print("⚖️  MODEL COMPARISON & SELECTION")
        print("=" * 60)

        # Create comparison dataframe
        comparison_df = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                "Logistic Regression": [
                    self.lr_metrics["accuracy"],
                    self.lr_metrics["precision"],
                    self.lr_metrics["recall"],
                    self.lr_metrics["f1_score"],
                    self.lr_metrics["roc_auc"],
                ],
                "Decision Tree": [
                    self.dt_metrics["accuracy"],
                    self.dt_metrics["precision"],
                    self.dt_metrics["recall"],
                    self.dt_metrics["f1_score"],
                    self.dt_metrics["roc_auc"],
                ],
            }
        )

        print("\n  📊 PERFORMANCE COMPARISON:")
        print(comparison_df.to_string(index=False))

        # Determine best model
        lr_score = self.lr_metrics["f1_score"]
        dt_score = self.dt_metrics["f1_score"]

        if lr_score > dt_score:
            best_model = "Logistic Regression"
            improvement = lr_score - dt_score
        else:
            best_model = "Decision Tree"
            improvement = dt_score - lr_score

        print(f"\n  🏆 BEST MODEL: {best_model}")
        print(f"     F1-Score advantage: {improvement:.4f}")

        self.best_model_name = best_model
        self.evaluation_results["best_model"] = best_model
        self.evaluation_results["comparison"] = comparison_df

        return comparison_df

    # ==================== VISUALIZATION SECTION ====================

    def plot_confusion_matrices(self, save=True):
        """Generate confusion matrix heatmaps for both models"""
        print("\n" + "=" * 60)
        print("🎨 GENERATING CONFUSION MATRIX VISUALIZATIONS")
        print("=" * 60)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "Confusion Matrices - Model Comparison", fontsize=14, fontweight="bold"
        )

        # Logistic Regression
        sns.heatmap(
            self.lr_metrics["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[0],
            cbar=False,
            xticklabels=self.target_labels,
            yticklabels=self.target_labels,
        )
        axes[0].set_title("Logistic Regression", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("True Label")
        axes[0].set_xlabel("Predicted Label")

        # Decision Tree
        sns.heatmap(
            self.dt_metrics["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Greens",
            ax=axes[1],
            cbar=False,
            xticklabels=self.target_labels,
            yticklabels=self.target_labels,
        )
        axes[1].set_title("Decision Tree", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("True Label")
        axes[1].set_xlabel("Predicted Label")

        plt.tight_layout()

        if save:
            save_path = config.OUTPUTS_DIR / "plots" / "confusion_matrices.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  ✅ Confusion matrices saved: {save_path}")

        return fig

    def plot_roc_curves(self, save=True):
        """Generate ROC curves for both models"""
        print("\n" + "=" * 60)
        print("🎨 GENERATING ROC CURVES")
        print("=" * 60)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")

        # Logistic Regression
        fpr_lr, tpr_lr, _ = roc_curve(self.y_test, self.lr_metrics["y_pred_proba"])
        auc_lr = self.lr_metrics["roc_auc"]
        axes[0].plot(
            fpr_lr,
            tpr_lr,
            color="blue",
            lw=2,
            label=f"ROC curve (AUC = {auc_lr:.4f})",
        )
        axes[0].plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("Logistic Regression", fontsize=12, fontweight="bold")
        axes[0].legend(loc="lower right")
        axes[0].grid(alpha=0.3)

        # Decision Tree
        fpr_dt, tpr_dt, _ = roc_curve(self.y_test, self.dt_metrics["y_pred_proba"])
        auc_dt = self.dt_metrics["roc_auc"]
        axes[1].plot(
            fpr_dt,
            tpr_dt,
            color="green",
            lw=2,
            label=f"ROC curve (AUC = {auc_dt:.4f})",
        )
        axes[1].plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("Decision Tree", fontsize=12, fontweight="bold")
        axes[1].legend(loc="lower right")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = config.OUTPUTS_DIR / "plots" / "roc_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  ✅ ROC curves saved: {save_path}")

        return fig

    def plot_feature_importance_comparison(self, top_n=15, save=True):
        """Compare feature importance between models"""
        print("\n" + "=" * 60)
        print("🎨 GENERATING FEATURE IMPORTANCE COMPARISON")
        print("=" * 60)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data
        lr_top = self.lr_importance.head(top_n).copy()
        lr_top["model"] = "Logistic Regression"

        dt_top = self.dt_importance.head(top_n).copy()
        dt_top["model"] = "Decision Tree"

        # Plot top features from LR
        y_pos = np.arange(len(lr_top))
        ax.barh(
            y_pos - 0.2,
            lr_top[
                "abs_coefficient" if "coefficient" in lr_top.columns else "importance"
            ],
            0.4,
            label="Logistic Regression",
            color="steelblue",
            alpha=0.8,
        )

        # Plot top features from DT
        dt_top_reindexed = dt_top.reindex(lr_top.index, fill_value=0)
        ax.barh(
            y_pos + 0.2,
            dt_top_reindexed["importance"],
            0.4,
            label="Decision Tree",
            color="seagreen",
            alpha=0.8,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(lr_top["feature"].values, fontsize=9)
        ax.set_xlabel("Importance Score")
        ax.set_title(
            "Top Features Comparison - Logistic Regression vs Decision Tree",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(alpha=0.3, axis="x")

        plt.tight_layout()

        if save:
            save_path = (
                config.OUTPUTS_DIR / "plots" / "feature_importance_comparison.png"
            )
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  ✅ Feature importance comparison saved: {save_path}")

        return fig

    def plot_metrics_comparison_bar(self, save=True):
        """Bar plot comparing all metrics"""
        print("\n" + "=" * 60)
        print("🎨 GENERATING METRICS COMPARISON CHART")
        print("=" * 60)

        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        lr_values = [
            self.lr_metrics["accuracy"],
            self.lr_metrics["precision"],
            self.lr_metrics["recall"],
            self.lr_metrics["f1_score"],
            self.lr_metrics["roc_auc"],
        ]
        dt_values = [
            self.dt_metrics["accuracy"],
            self.dt_metrics["precision"],
            self.dt_metrics["recall"],
            self.dt_metrics["f1_score"],
            self.dt_metrics["roc_auc"],
        ]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            lr_values,
            width,
            label="Logistic Regression",
            color="steelblue",
            alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            dt_values,
            width,
            label="Decision Tree",
            color="seagreen",
            alpha=0.8,
        )

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_ylabel("Score")
        ax.set_title("Evaluation Metrics Comparison", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()

        if save:
            save_path = config.OUTPUTS_DIR / "plots" / "metrics_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  ✅ Metrics comparison chart saved: {save_path}")

        return fig

    # ==================== REPORT GENERATION ====================

    def generate_evaluation_report(self, output_path=None):
        """Generate comprehensive evaluation report"""
        if output_path is None:
            output_path = config.OUTPUTS_DIR / "reports" / "model_evaluation_report.txt"

        print("\n" + "=" * 60)
        print("📝 GENERATING EVALUATION REPORT")
        print("=" * 60)

        report = f"""
{'='*80}
COMPREHENSIVE MODEL EVALUATION REPORT
{'='*80}

SECTION 5.1: REQUIRED EVALUATION METRICS (COURSE SPECIFIED)

1. LOGISTIC REGRESSION MODEL
{'-'*80}
Hyperparameters: {self.lr_best_params}

Performance Metrics:
  • Accuracy:  {self.lr_metrics['accuracy']:.4f}
  • Precision: {self.lr_metrics['precision']:.4f}
  • Recall:    {self.lr_metrics['recall']:.4f}
  • F1-Score:  {self.lr_metrics['f1_score']:.4f}
  • ROC-AUC:   {self.lr_metrics['roc_auc']:.4f}

Confusion Matrix:
  {self.lr_metrics['confusion_matrix']}

Classification Report:
{self.lr_report}

2. DECISION TREE MODEL
{'-'*80}
Hyperparameters: {self.dt_best_params}

Performance Metrics:
  • Accuracy:  {self.dt_metrics['accuracy']:.4f}
  • Precision: {self.dt_metrics['precision']:.4f}
  • Recall:    {self.dt_metrics['recall']:.4f}
  • F1-Score:  {self.dt_metrics['f1_score']:.4f}
  • ROC-AUC:   {self.dt_metrics['roc_auc']:.4f}

Confusion Matrix:
  {self.dt_metrics['confusion_matrix']}

Classification Report:
{self.dt_report}

3. MODEL COMPARISON SUMMARY
{'-'*80}
Best Performing Model: {self.best_model_name}

Performance Comparison:
{self.evaluation_results['comparison'].to_string(index=False)}

4. SECTION 5.1.2: PERFORMANCE VISUALIZATIONS GENERATED
{'-'*80}
✅ Confusion matrix heatmaps (confusion_matrices.png)
✅ ROC curves for both models (roc_curves.png)
✅ Feature importance comparison (feature_importance_comparison.png)
✅ Metrics comparison bar chart (metrics_comparison.png)

5. ACADEMIC EVALUATION NOTES
{'-'*80}
✅ All course-required metrics calculated (Accuracy, Precision, Recall, F1)
✅ Confusion matrices generated for visual evaluation
✅ ROC curves created for model comparison
✅ Feature importance extracted from both models
✅ Models properly compared and best model identified
✅ Visualizations saved for presentation

{'='*80}
SECTION 5 EVALUATION COMPLETE
Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

        with open(output_path, "w") as f:
            f.write(report)

        print(f"  ✅ Evaluation report saved: {output_path}")
        return output_path

    def run_complete_evaluation(self):
        """Execute complete model evaluation pipeline"""
        print("\n" + "=" * 80)
        print("🎯 RUNNING COMPLETE MODEL EVALUATION (SECTION 5)")
        print("=" * 80)

        # Calculate metrics
        self.calculate_all_metrics()
        self.generate_classification_reports()
        self.compare_and_select_best_model()

        # Generate visualizations
        self.plot_confusion_matrices(save=True)
        self.plot_roc_curves(save=True)
        self.plot_feature_importance_comparison(save=True)
        self.plot_metrics_comparison_bar(save=True)

        # Generate report
        self.generate_evaluation_report()

        print("\n" + "=" * 80)
        print("✅ SECTION 5: MODEL EVALUATION COMPLETE")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"  • Logistic Regression F1: {self.lr_metrics['f1_score']:.4f}")
        print(f"  • Decision Tree F1: {self.dt_metrics['f1_score']:.4f}")
        print(f"  • Best Model: {self.best_model_name}")
        print(f"  • Visualizations saved: {config.OUTPUTS_DIR / 'plots'}")
        print(
            f"  • Report saved: {config.OUTPUTS_DIR / 'reports' / 'model_evaluation_report.txt'}"
        )

        return self.evaluation_results


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    results = evaluator.run_complete_evaluation()
