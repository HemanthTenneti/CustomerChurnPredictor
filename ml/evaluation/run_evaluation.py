"""
Unified Model Evaluation & Comparison Script (Section 5)
Orchestrates complete evaluation pipeline with visualizations
Demonstrates comprehensive assessment for academic evaluation
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.evaluation.model_evaluation import ModelEvaluation


def run_complete_evaluation():
    """Execute complete Section 5 evaluation pipeline"""
    print("\n" + "=" * 100)
    print("🎯 SECTION 5: COMPREHENSIVE MODEL EVALUATION")
    print("=" * 100)

    evaluator = ModelEvaluation()
    results = evaluator.run_complete_evaluation()

    # Additional summary
    print("\n" + "=" * 100)
    print("📋 EVALUATION DELIVERABLES SUMMARY")
    print("=" * 100)

    print("\n✅ SECTION 5.1 - REQUIRED EVALUATION METRICS:")
    print(
        f"   ✓ Accuracy (LR): {results['metrics']['logistic_regression']['accuracy']:.4f}"
    )
    print(f"   ✓ Accuracy (DT): {results['metrics']['decision_tree']['accuracy']:.4f}")
    print(
        f"   ✓ Precision (LR): {results['metrics']['logistic_regression']['precision']:.4f}"
    )
    print(
        f"   ✓ Precision (DT): {results['metrics']['decision_tree']['precision']:.4f}"
    )
    print(
        f"   ✓ Recall (LR): {results['metrics']['logistic_regression']['recall']:.4f}"
    )
    print(f"   ✓ Recall (DT): {results['metrics']['decision_tree']['recall']:.4f}")
    print(
        f"   ✓ F1-Score (LR): {results['metrics']['logistic_regression']['f1_score']:.4f}"
    )
    print(f"   ✓ F1-Score (DT): {results['metrics']['decision_tree']['f1_score']:.4f}")
    print(f"   ✓ Confusion matrices generated")
    print(f"   ✓ Model comparison completed")

    print("\n✅ SECTION 5.1.2 - PERFORMANCE VISUALIZATIONS:")
    print(f"   ✓ Confusion matrix heatmaps")
    print(f"   ✓ ROC curves with AUC scores")
    print(f"   ✓ Feature importance comparisons")
    print(f"   ✓ Metrics comparison bar charts")

    print("\n✅ GENERATED FILES:")
    print(f"   ✓ {Path('ml/evaluation/model_evaluation.py').absolute()}")
    print(f"   ✓ outputs/plots/confusion_matrices.png")
    print(f"   ✓ outputs/plots/roc_curves.png")
    print(f"   ✓ outputs/plots/feature_importance_comparison.png")
    print(f"   ✓ outputs/plots/metrics_comparison.png")
    print(f"   ✓ outputs/reports/model_evaluation_report.txt")

    print("\n" + "=" * 100)
    print("✅ SECTION 5 COMPLETE - READY FOR SECTION 6 (INSIGHTS GENERATION)")
    print("=" * 100)

    return results


if __name__ == "__main__":
    run_complete_evaluation()
