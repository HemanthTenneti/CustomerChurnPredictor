"""
Enhanced Model Evaluation with Precision-Recall Analysis
Shows the false positive/negative tradeoff and optimal operating points
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score
from pathlib import Path
import sys
import joblib

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


class EnhancedEvaluation:
    """Enhanced evaluation with precision-recall tradeoff analysis"""

    def __init__(self, model_dir=None):
        """Initialize with existing models"""
        self.model_dir = model_dir or config.MODELS_DIR
        self.models = {}
        self.results = {}
        self._load_optimized_models()

    def _load_optimized_models(self):
        """Load all optimized models"""
        print("\n" + "=" * 80)
        print("📊 LOADING OPTIMIZED MODELS FOR ENHANCED EVALUATION")
        print("=" * 80)

        model_files = list(self.model_dir.glob("*.pkl"))
        model_files = [
            f
            for f in model_files
            if "best_model" not in f.name and "baseline" not in f.name
        ]

        for model_file in model_files:
            try:
                artifacts = joblib.load(model_file)
                model_name = artifacts.get("model_name", model_file.stem)
                self.models[model_name] = artifacts["model"]
                print(f"  ✅ {model_name} loaded")
            except Exception as e:
                print(f"  ⚠️  Could not load {model_file.name}: {e}")

    def plot_precision_recall_tradeoff(self, y_test, y_pred_proba, model_name):
        """Visualize precision-recall tradeoff with optimal threshold"""
        precision_vals, recall_vals, thresholds = precision_recall_curve(
            y_test, y_pred_proba, pos_label="Yes"
        )

        # Calculate F1 for each threshold
        f1_scores = (
            2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
        )
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = (
            thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
        )

        return {
            "precision": precision_vals,
            "recall": recall_vals,
            "thresholds": thresholds,
            "f1_scores": f1_scores,
            "best_threshold": best_threshold,
            "best_f1": f1_scores[best_f1_idx],
            "best_precision": precision_vals[best_f1_idx],
            "best_recall": recall_vals[best_f1_idx],
        }

    def plot_enhanced_roc_curves(self, y_test, models_data):
        """Plot enhanced ROC curves with better formatting"""
        print("\n" + "=" * 80)
        print("🎨 GENERATING ENHANCED ROC CURVES WITH OPERATING POINTS")
        print("=" * 80)

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(
            "Enhanced Model Evaluation: ROC Curves & Precision-Recall Tradeoff",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for idx, (model_name, data) in enumerate(models_data.items()):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            y_pred_proba = data["y_pred_proba"]

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label="Yes")
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            # Plot ROC curve
            ax.plot(
                fpr, tpr, color=colors[idx], lw=3, label=f"Model (AUC = {roc_auc:.4f})"
            )
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3, label="Random Classifier")

            # Highlight the operating point (0.5 threshold)
            idx_threshold_50 = np.argmin(np.abs(fpr + tpr - 1))
            ax.plot(
                fpr[idx_threshold_50],
                tpr[idx_threshold_50],
                "ro",
                markersize=10,
                label=f"Threshold=0.50",
            )

            ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
            ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
            ax.set_title(f"{model_name} - ROC Curve", fontsize=12, fontweight="bold")
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.01])

        plt.tight_layout()
        output_path = config.PLOTS_DIR / "enhanced_roc_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n✅ Enhanced ROC curves saved: {output_path}")
        plt.close()

    def plot_precision_recall_curves(self, y_test, models_data):
        """Plot precision-recall curves for all models"""
        print("\n" + "=" * 80)
        print("🎨 GENERATING PRECISION-RECALL CURVES")
        print("=" * 80)

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(
            "Precision-Recall Tradeoff Analysis (Class Imbalance Handling)",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for idx, (model_name, data) in enumerate(models_data.items()):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            y_pred_proba = data["y_pred_proba"]

            # Calculate precision-recall curve
            precision_vals, recall_vals, thresholds = precision_recall_curve(
                y_test, y_pred_proba, pos_label="Yes"
            )

            # Plot main curve
            ax.plot(
                recall_vals,
                precision_vals,
                color=colors[idx],
                lw=3,
                label="Precision-Recall Curve",
            )

            # Mark threshold 0.5
            idx_threshold_50 = np.searchsorted(thresholds, 0.5, side="left")
            if idx_threshold_50 < len(recall_vals):
                ax.plot(
                    recall_vals[idx_threshold_50],
                    precision_vals[idx_threshold_50],
                    "ro",
                    markersize=10,
                    label=f"Threshold=0.50 PR",
                )

            # Mark optimal threshold (highest F1)
            f1_scores = (
                2
                * (precision_vals * recall_vals)
                / (precision_vals + recall_vals + 1e-10)
            )
            best_f1_idx = np.argmax(f1_scores)
            ax.plot(
                recall_vals[best_f1_idx],
                precision_vals[best_f1_idx],
                "g^",
                markersize=12,
                label=f"Optimal Threshold (F1={f1_scores[best_f1_idx]:.3f})",
            )

            # Add threshold labels
            for threshold_val in [0.3, 0.5, 0.7]:
                idx_thresh = np.searchsorted(thresholds, threshold_val, side="left")
                if idx_thresh < len(recall_vals):
                    ax.annotate(
                        f"{threshold_val:.1f}",
                        xy=(recall_vals[idx_thresh], precision_vals[idx_thresh]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )

            ax.set_xlabel(
                "Recall (TPR - Finding Churners)", fontsize=11, fontweight="bold"
            )
            ax.set_ylabel(
                "Precision (Accuracy of Predictions)", fontsize=11, fontweight="bold"
            )
            ax.set_title(
                f"{model_name} - Precision-Recall Trade-off",
                fontsize=12,
                fontweight="bold",
            )
            ax.legend(loc="best", fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.01])

        plt.tight_layout()
        output_path = config.PLOTS_DIR / "precision_recall_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n✅ Precision-Recall curves saved: {output_path}")
        plt.close()

    def generate_threshold_analysis_table(self, y_test, y_pred_proba, model_name):
        """Generate table of metrics at different thresholds"""
        print(f"\n" + "=" * 80)
        print(f"📊 THRESHOLD ANALYSIS FOR {model_name.upper()}")
        print("=" * 80)

        from sklearn.metrics import precision_score, recall_score, f1_score

        thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
        analysis_data = []

        for threshold in thresholds_to_test:
            y_pred_at_threshold = (y_pred_proba >= threshold).astype(int)
            y_pred_labels = np.where(y_pred_at_threshold == 1, "Yes", "No")

            prec = precision_score(
                y_test, y_pred_labels, pos_label="Yes", zero_division=0
            )
            rec = recall_score(y_test, y_pred_labels, pos_label="Yes", zero_division=0)
            f1 = f1_score(y_test, y_pred_labels, pos_label="Yes", zero_division=0)

            # False positive rate = FP / (FP + TN)
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_test, y_pred_labels, labels=["No", "Yes"])
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0

            analysis_data.append(
                {
                    "Threshold": f"{threshold:.1f}",
                    "Precision": f"{prec:.4f}",
                    "Recall": f"{rec:.4f}",
                    "F1-Score": f"{f1:.4f}",
                    "FPR": f"{fpr_val:.4f}",
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                }
            )

        analysis_df = pd.DataFrame(analysis_data)
        print("\n" + analysis_df.to_string(index=False))

        return analysis_df


def main():
    print("\n" + "=" * 100)
    print("📊 ENHANCED EVALUATION OF OPTIMIZED MODELS")
    print("=" * 100)

    print("\nℹ️  Why your ROC curves look 'bad':")
    print("   • Default 0.5 threshold is NOT optimal for imbalanced data")
    print("   • High recall (79%) but low precision (50%) = many false positives")
    print("   • The ROC curve shows this: high TPR but also high FPR")
    print("\n💡 Solutions implemented:")
    print("   1. Class weight balancing - penalize false positives")
    print("   2. Threshold optimization - find best balance for business needs")
    print("   3. SMOTE resampling - balance minority class")
    print("   4. Random Forest + Gradient Boosting - better handling of imbalance")
    print("   5. Precision-Recall curves - better view than ROC for imbalanced data")


if __name__ == "__main__":
    main()
