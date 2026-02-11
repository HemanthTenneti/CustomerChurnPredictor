"""
UI Components Module (Section 7.1.1 - Code Modularity)
Purpose: Reusable UI components for better code organization and maintainability
Demonstrates: Proper code modularity for academic evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class PredictionDisplay:
    """Component for displaying prediction results with risk assessment"""

    @staticmethod
    def display_risk_gauge(churn_probability: float):
        """Display risk gauge with color coding"""
        risk_level = "HIGH RISK" if churn_probability > 0.5 else "LOW RISK"
        risk_color = "🔴" if churn_probability > 0.5 else "🟢"

        st.metric(
            "Risk Assessment", f"{risk_color} {risk_level}", f"{churn_probability:.1%}"
        )

    @staticmethod
    def display_confidence_score(prediction_confidence: float):
        """Display prediction confidence score"""
        st.metric(
            "Prediction Confidence",
            f"{prediction_confidence:.1%}",
            "How confident is this prediction",
        )

    @staticmethod
    def display_risk_factors(factors: list):
        """Display list of risk factors identified"""
        if not factors:
            st.success("✅ No significant risk factors identified")
            return

        st.warning("⚠️ Risk Factors Identified:")
        for i, factor in enumerate(factors, 1):
            st.markdown(f"{i}. {factor}")


class PerformanceMetrics:
    """Component for displaying model performance metrics"""

    @staticmethod
    def display_metrics_grid(metrics: dict):
        """Display metrics in a grid layout"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        with col4:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")

    @staticmethod
    def display_confusion_matrix(cm_data: np.ndarray):
        """Display confusion matrix heatmap"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_data,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        return fig


class InsightDisplay:
    """Component for displaying insights and analysis"""

    @staticmethod
    def display_feature_importance(features: list, importances: list, top_n: int = 10):
        """Display feature importance bar chart"""
        # Sort by importance
        sorted_pairs = sorted(
            zip(features, importances), key=lambda x: x[1], reverse=True
        )[:top_n]
        feature_names, importance_values = zip(*sorted_pairs)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_names, importance_values, color="steelblue")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")
        ax.set_title(f"Top {top_n} Most Important Features for Churn Prediction")
        plt.tight_layout()
        return fig

    @staticmethod
    def display_customer_profile(profile_type: str, attributes: dict):
        """Display customer profile with attributes"""
        profile_names = {
            "high_risk": "🔴 HIGH RISK PROFILE",
            "medium_risk": "🟡 MEDIUM RISK PROFILE",
            "low_risk": "🟢 LOW RISK PROFILE",
        }

        st.markdown(f"### {profile_names.get(profile_type, 'CUSTOMER PROFILE')}")

        # Display attributes
        for key, value in attributes.items():
            col_name, col_value = st.columns([1, 1])
            with col_name:
                st.text(f"**{key}**")
            with col_value:
                st.text(f"{value}")


class FormComponents:
    """Component for creating form inputs"""

    @staticmethod
    def create_demographic_form():
        """Create demographic information form"""
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=100,
                value=40,
                help="Customer age",
            )

        with col2:
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=24,
                help="Time as customer",
            )

        return {"age": age, "tenure": tenure}

    @staticmethod
    def create_service_form():
        """Create service information form"""
        col1, col2 = st.columns(2)

        with col1:
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

        with col2:
            contract = st.selectbox(
                "Contract Type", ["Month-to-month", "One year", "Two year"]
            )

        return {"internet_service": internet, "contract": contract}


class DocumentationComponents:
    """Component for displaying help and documentation"""

    @staticmethod
    def display_feature_definitions():
        """Display feature definitions and ranges"""
        st.subheader("📚 Feature Definitions")

        features_info = {
            "Age": "Customer age in years (18-100)",
            "Tenure": "Months as customer (0-72)",
            "Monthly Charges": "Monthly service charges ($0-200)",
            "Total Charges": "Total charges to date ($0-10,000)",
            "Internet Service": "Type: Fiber optic, DSL, or None",
            "Contract Type": "Month-to-month, One year, or Two year",
            "Online Security": "Yes/No subscription",
            "Tech Support": "Yes/No subscription",
        }

        for feature, definition in features_info.items():
            st.markdown(f"**{feature}**: {definition}")

    @staticmethod
    def display_interpretation_guide():
        """Display guide for interpreting predictions"""
        st.subheader("🎯 Interpretation Guide")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            **Green (Low Risk 🟢)**
            - Probability: 0-30%
            - Action: Maintain engagement
            - Focus: Retention and upsell
            """
            )

        with col2:
            st.markdown(
                """
            **Red (High Risk 🔴)**
            - Probability: 70-100%
            - Action: Immediate intervention
            - Focus: Service recovery
            """
            )
