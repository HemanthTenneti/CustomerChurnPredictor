"""
Unified Insights Generation Script (Section 6)
Orchestrates complete insights pipeline for business understanding
Prepares data for UI integration
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.insights.basic_insights import ChurnInsights


def run_insights_generation():
    """Execute complete Section 6 insights generation pipeline"""
    print("\n" + "=" * 100)
    print("💡 SECTION 6: BASIC INSIGHTS GENERATION")
    print("=" * 100)

    insights_gen = ChurnInsights()
    ui_data = insights_gen.run_complete_insights_generation()

    # Display summary
    print("\n" + "=" * 100)
    print("📋 INSIGHTS DELIVERABLES SUMMARY")
    print("=" * 100)

    print("\n✅ SECTION 6.1.1 - FEATURE IMPORTANCE ANALYSIS:")
    for idx, (_, feature) in enumerate(
        insights_gen.top_features.head(10).iterrows(), 1
    ):
        print(
            f"   {idx:2d}. {feature['feature']:<40} (importance: {feature['importance']:.4f})"
        )

    print("\n✅ SECTION 6.1.1 - FEATURE INTERPRETATIONS:")
    for idx, (feature, info) in enumerate(
        list(insights_gen.interpretations.items())[:5], 1
    ):
        print(f"   {idx}. {feature}: {info['explanation']}")

    print("\n✅ SECTION 6.1.2 - HIGH-RISK CUSTOMER SEGMENTS:")
    print(f"   • High-risk customers: {len(insights_gen.high_risk_indices)}")
    print(f"   • Low-risk customers: {len(insights_gen.low_risk_indices)}")
    print(
        f"   • Key characteristics identified: {list(insights_gen.high_risk_profile.keys())}"
    )

    print("\n✅ SECTION 6.1.2 - CHURN PATTERN INSIGHTS:")
    for feature, patterns in insights_gen.churn_patterns.items():
        print(
            f"   • {feature}: Churned avg {patterns['churned_avg']:.2f} vs Retained {patterns['retained_avg']:.2f}"
        )

    print("\n✅ ACTIONABLE BUSINESS INSIGHTS:")
    for insight in insights_gen.actionable_insights:
        print(f"   • {insight}")

    print("\n✅ VISUALIZATIONS GENERATED:")
    print(f"   ✓ Churn drivers (feature importance)")
    print(f"   ✓ Risk distribution (customer segments)")

    print("\n✅ GENERATED FILES:")
    print(f"   ✓ ml/insights/basic_insights.py")
    print(f"   ✓ outputs/plots/churn_drivers.png")
    print(f"   ✓ outputs/plots/risk_distribution.png")
    print(f"   ✓ outputs/reports/churn_insights_report.txt")

    print("\n✅ UI-READY DATA STRUCTURE:")
    print(f"   ✓ Top features (10): {len(ui_data['top_features'])} items")
    print(f"   ✓ Interpretations (5): {len(ui_data['interpretations'])} items")
    print(f"   ✓ Risk profile: {list(ui_data['high_risk_profile'].keys())}")
    print(f"   ✓ Actionable insights: {len(ui_data['actionable_insights'])} items")

    print("\n" + "=" * 100)
    print("✅ SECTION 6 COMPLETE - READY FOR SECTION 7 (UI DEVELOPMENT)")
    print("=" * 100)

    return ui_data


if __name__ == "__main__":
    run_insights_generation()
