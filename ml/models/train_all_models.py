import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.models.baseline_models import BaselineModels
from ml.models.course_models import CourseModels


def train_all_models():
    """Train all baseline and course-required models"""
    print("\n" + "=" * 100)
    print("🚀 SECTION 4: COMPLETE MODEL TRAINING PIPELINE")
    print("=" * 100)

    # ============== BASELINE MODELS ==============
    print("\n" + "▓" * 100)
    print("▓" + " " * 98 + "▓")
    print("▓" + "SECTION 4.1: BASELINE MODEL TRAINING".center(98) + "▓")
    print("▓" + " " * 98 + "▓")
    print("▓" * 100)

    baseline = BaselineModels()
    baseline_results = baseline.run_baseline_training()

    # ============== COURSE MODELS ==============
    print("\n" + "▓" * 100)
    print("▓" + " " * 98 + "▓")
    print("▓" + "SECTION 4.2: COURSE-REQUIRED MODEL TRAINING".center(98) + "▓")
    print("▓" + " " * 98 + "▓")
    print("▓" * 100)

    course_models = CourseModels()
    course_results = course_models.run_course_model_training()

    # ============== SUMMARY ==============
    print("\n" + "=" * 100)
    print("📊 SECTION 4 TRAINING SUMMARY")
    print("=" * 100)

    print("\n🔹 BASELINE MODEL (4.1):")
    print(f"   ✅ Logistic Regression (default params)")
    print(
        f"   📈 Accuracy: {baseline_results['accuracy']:.4f} | F1: {baseline_results['f1_score']:.4f}"
    )

    print("\n🔹 COURSE MODELS (4.2):")
    print(f"   ✅ Logistic Regression (tuned)")
    print(
        f"   📈 Accuracy: {course_results['logistic_regression']['accuracy']:.4f} | F1: {course_results['logistic_regression']['f1_score']:.4f}"
    )
    print(f"   ✅ Decision Tree (tuned)")
    print(
        f"   📈 Accuracy: {course_results['decision_tree']['accuracy']:.4f} | F1: {course_results['decision_tree']['f1_score']:.4f}"
    )
    print(f"   🏆 Best Model: {course_results['best_model']}")

    print("\n🔹 FILES GENERATED:")
    print(f"   ✅ /ml/models/baseline_logistic_regression.pkl")
    print(f"   ✅ /ml/models/baseline_metrics.txt")
    print(f"   ✅ /ml/models/logistic_regression_tuned.pkl")
    print(f"   ✅ /ml/models/decision_tree_tuned.pkl")
    print(
        f"   ✅ /ml/models/best_model_{course_results['best_model'].lower().replace(' ', '_')}.pkl"
    )

    print("\n" + "=" * 100)
    print("✅ SECTION 4 COMPLETE - READY FOR SECTION 5 (MODEL EVALUATION)")
    print("=" * 100)


if __name__ == "__main__":
    train_all_models()
