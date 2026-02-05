"""
Data Quality Assessment for Customer Churn Prediction
Quantify data quality issues requiring preprocessing solutions
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


class DataQualityAssessment:
    """
    Comprehensive data quality assessment for academic evaluation
    Identifies and quantifies data quality issues for preprocessing planning
    """

    def __init__(self, data_path=None):
        """Initialize with dataset"""
        if data_path is None:
            data_path = config.CHURN_DATA_PATH

        print("🔍 Loading dataset for quality assessment...")
        self.df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded: {self.df.shape}")

        # Ensure validation directory exists
        config.create_directories()

    def missing_data_analysis(self):
        """Comprehensive missing data analysis and strategy recommendation"""
        print("\n" + "=" * 60)
        print("🕳️  MISSING DATA ANALYSIS")
        print("=" * 60)

        missing_analysis = {}

        # Calculate missing values per column
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100

        print("Missing Data Summary:")
        print(f"{'Column':<20} {'Missing Count':<15} {'Percentage':<12} {'Strategy'}")
        print("-" * 65)

        for col in self.df.columns:
            count = missing_counts[col]
            pct = missing_percentages[col]

            # Determine missing data strategy based on percentage and type
            if pct == 0:
                strategy = "No action needed"
            elif pct < 5:
                if self.df[col].dtype == "object":
                    strategy = "Mode imputation"
                else:
                    strategy = "Mean/median imputation"
            elif pct < 15:
                strategy = "Advanced imputation"
            else:
                strategy = "Consider dropping or new category"

            print(f"{col:<20} {count:<15} {pct:<12.2f} {strategy}")

            missing_analysis[col] = {
                "missing_count": int(count),
                "missing_percentage": float(pct),
                "recommended_strategy": strategy,
            }

        # Special handling for TotalCharges (common issue)
        if "TotalCharges" in self.df.columns:
            print(f"\n🔍 Special Analysis: TotalCharges")

            # Check for string issues
            if self.df["TotalCharges"].dtype == "object":
                non_numeric = self.df[self.df["TotalCharges"].str.strip() == " "]
                print(f"  Empty string values: {len(non_numeric)}")

                if len(non_numeric) > 0:
                    print(
                        f"  Strategy: Convert empty strings to NaN, then impute using tenure × MonthlyCharges"
                    )
                    missing_analysis["TotalCharges"]["special_issue"] = "empty_strings"
                    missing_analysis["TotalCharges"][
                        "recommended_strategy"
                    ] = "tenure_monthly_imputation"

        return missing_analysis

    def data_type_consistency_analysis(self):
        """Analyze data type consistency and conversion needs"""
        print("\n" + "=" * 60)
        print("🔢 DATA TYPE CONSISTENCY ANALYSIS")
        print("=" * 60)

        type_analysis = {}

        print(f"{'Column':<20} {'Current Type':<15} {'Expected Type':<15} {'Issues'}")
        print("-" * 70)

        for col in self.df.columns:
            current_type = str(self.df[col].dtype)
            issues = []
            expected_type = current_type  # Default to current

            # Analyze specific columns with known patterns
            if col == "TotalCharges":
                expected_type = "float64"
                if current_type == "object":
                    issues.append("Should be numeric")

            elif col == "SeniorCitizen":
                expected_type = "int64"
                if current_type not in ["int64", "int32"]:
                    issues.append("Should be integer (0/1)")

            elif col in ["MonthlyCharges", "tenure"]:
                expected_type = "float64" if col == "MonthlyCharges" else "int64"
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    issues.append("Should be numeric")

            # Check for categorical variables that should be categories
            elif self.df[col].dtype == "object" and self.df[col].nunique() < 20:
                expected_type = "category"

            issues_str = "; ".join(issues) if issues else "None"
            print(f"{col:<20} {current_type:<15} {expected_type:<15} {issues_str}")

            type_analysis[col] = {
                "current_type": current_type,
                "expected_type": expected_type,
                "issues": issues,
                "conversion_needed": len(issues) > 0,
            }

        return type_analysis

    def outlier_detection_analysis(self):
        """Comprehensive outlier detection using multiple methods"""
        print("\n" + "=" * 60)
        print("📊 OUTLIER DETECTION ANALYSIS")
        print("=" * 60)

        outlier_analysis = {}

        # Get numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Handle TotalCharges conversion if needed
        df_numeric = self.df.copy()
        if (
            "TotalCharges" in df_numeric.columns
            and df_numeric["TotalCharges"].dtype == "object"
        ):
            df_numeric["TotalCharges"] = pd.to_numeric(
                df_numeric["TotalCharges"], errors="coerce"
            )
            numerical_cols = df_numeric.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        print(
            f"{'Column':<15} {'IQR Outliers':<15} {'Z-Score Outliers':<18} {'Recommendation'}"
        )
        print("-" * 75)

        for col in numerical_cols:
            data = df_numeric[col].dropna()

            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
            iqr_count = len(iqr_outliers)
            iqr_pct = (iqr_count / len(data)) * 100

            # Z-Score method (|z| > 3)
            z_scores = np.abs((data - data.mean()) / data.std())
            zscore_outliers = data[z_scores > 3]
            zscore_count = len(zscore_outliers)
            zscore_pct = (zscore_count / len(data)) * 100

            # Determine recommendation
            if max(iqr_pct, zscore_pct) > 5:
                recommendation = "Cap/transform"
            elif max(iqr_pct, zscore_pct) > 2:
                recommendation = "Investigate"
            else:
                recommendation = "Keep as-is"

            print(
                f"{col:<15} {iqr_count} ({iqr_pct:.1f}%)<12 {zscore_count} ({zscore_pct:.1f}%)<16 {recommendation}"
            )

            outlier_analysis[col] = {
                "iqr_outliers": int(iqr_count),
                "iqr_percentage": float(iqr_pct),
                "zscore_outliers": int(zscore_count),
                "zscore_percentage": float(zscore_pct),
                "recommendation": recommendation,
                "outlier_values_iqr": iqr_outliers.tolist()[
                    :10
                ],  # Store up to 10 examples
            }

        return outlier_analysis

    def business_logic_validation(self):
        """Validate data consistency with telecommunications domain knowledge"""
        print("\n" + "=" * 60)
        print("💼 BUSINESS LOGIC VALIDATION")
        print("=" * 60)

        validation_results = {}

        # Convert TotalCharges for analysis
        df_clean = self.df.copy()
        if (
            "TotalCharges" in df_clean.columns
            and df_clean["TotalCharges"].dtype == "object"
        ):
            df_clean["TotalCharges"] = pd.to_numeric(
                df_clean["TotalCharges"], errors="coerce"
            )

        print("Business Rule Validations:")

        # 1. Tenure vs TotalCharges relationship
        if (
            "tenure" in df_clean.columns
            and "TotalCharges" in df_clean.columns
            and "MonthlyCharges" in df_clean.columns
        ):
            valid_data = df_clean[["tenure", "TotalCharges", "MonthlyCharges"]].dropna()

            # Approximate relationship: TotalCharges ≈ tenure × MonthlyCharges
            expected_total = valid_data["tenure"] * valid_data["MonthlyCharges"]
            actual_total = valid_data["TotalCharges"]

            # Allow for reasonable variation (±20%)
            tolerance = 0.20
            inconsistent = abs(actual_total - expected_total) > (
                expected_total * tolerance
            )
            inconsistent_count = inconsistent.sum()
            inconsistent_pct = (inconsistent_count / len(valid_data)) * 100

            print(
                f"  1. Tenure-TotalCharges consistency: {inconsistent_count} inconsistent ({inconsistent_pct:.1f}%)"
            )

            validation_results["tenure_totalcharges"] = {
                "inconsistent_count": int(inconsistent_count),
                "inconsistent_percentage": float(inconsistent_pct),
                "validation": "PASS" if inconsistent_pct < 5 else "WARNING",
            }

        # 2. Contract type vs tenure patterns
        if "Contract" in df_clean.columns and "tenure" in df_clean.columns:
            contract_tenure = df_clean.groupby("Contract")["tenure"].agg(
                ["mean", "median", "min", "max"]
            )
            print(f"  2. Contract vs Tenure patterns:")
            for contract_type, stats in contract_tenure.iterrows():
                print(
                    f"     {contract_type}: avg={stats['mean']:.1f}, range={stats['min']}-{stats['max']}"
                )

            validation_results["contract_tenure"] = {
                "patterns": contract_tenure.to_dict()
            }

        # 3. Service dependencies (streaming requires internet)
        streaming_cols = [col for col in df_clean.columns if "Streaming" in col]
        if streaming_cols and "InternetService" in df_clean.columns:
            no_internet = df_clean["InternetService"] == "No"

            for stream_col in streaming_cols:
                invalid_streaming = (no_internet) & (df_clean[stream_col] == "Yes")
                invalid_count = invalid_streaming.sum()

                if invalid_count > 0:
                    print(
                        f"  3. {stream_col} without internet: {invalid_count} cases (INVALID)"
                    )
                    validation_results[f"{stream_col}_dependency"] = {
                        "invalid_cases": int(invalid_count),
                        "validation": "FAIL",
                    }

        # 4. Range validations
        range_validations = {}

        # Tenure should be >= 0
        if "tenure" in df_clean.columns:
            invalid_tenure = df_clean["tenure"] < 0
            invalid_count = invalid_tenure.sum()
            range_validations["tenure_negative"] = int(invalid_count)
            if invalid_count > 0:
                print(f"  4. Negative tenure values: {invalid_count} cases (INVALID)")

        # MonthlyCharges should be > 0 for active customers
        if "MonthlyCharges" in df_clean.columns:
            invalid_charges = df_clean["MonthlyCharges"] <= 0
            invalid_count = invalid_charges.sum()
            range_validations["monthly_charges_zero"] = int(invalid_count)
            if invalid_count > 0:
                print(
                    f"  4. Zero/negative monthly charges: {invalid_count} cases (INVESTIGATE)"
                )

        validation_results["range_validations"] = range_validations

        return validation_results

    def generate_quality_report(self):
        """Generate comprehensive data quality report"""
        print("\n" + "=" * 60)
        print("📋 GENERATING DATA QUALITY REPORT")
        print("=" * 60)

        # Run all quality assessments
        missing_analysis = self.missing_data_analysis()
        type_analysis = self.data_type_consistency_analysis()
        outlier_analysis = self.outlier_detection_analysis()
        business_validation = self.business_logic_validation()

        # Compile comprehensive report
        quality_report = {
            "dataset_info": {
                "shape": self.df.shape,
                "memory_usage_mb": float(
                    self.df.memory_usage(deep=True).sum() / 1024**2
                ),
            },
            "missing_data_analysis": missing_analysis,
            "data_type_analysis": type_analysis,
            "outlier_analysis": outlier_analysis,
            "business_logic_validation": business_validation,
            "quality_score": self._calculate_quality_score(
                missing_analysis, type_analysis, outlier_analysis, business_validation
            ),
        }

        # Save report to JSON
        report_path = config.VALIDATION_DATA_DIR / "data_quality_metrics.json"
        with open(report_path, "w") as f:
            json.dump(quality_report, f, indent=2)

        print(f"✅ Quality report saved to: {report_path}")

        # Print summary
        self._print_quality_summary(quality_report)

        return quality_report

    def _calculate_quality_score(
        self, missing_analysis, type_analysis, outlier_analysis, business_validation
    ):
        """Calculate overall data quality score (0-100)"""
        score = 100.0

        # Deduct for missing data
        for col, analysis in missing_analysis.items():
            if analysis["missing_percentage"] > 0:
                score -= min(
                    analysis["missing_percentage"] * 0.5, 10
                )  # Max 10 points per column

        # Deduct for type issues
        type_issues = sum(
            1
            for col, analysis in type_analysis.items()
            if analysis["conversion_needed"]
        )
        score -= type_issues * 2  # 2 points per type issue

        # Deduct for excessive outliers
        for col, analysis in outlier_analysis.items():
            if analysis["iqr_percentage"] > 5:
                score -= 3  # 3 points for high outlier percentage

        # Deduct for business logic violations
        for key, validation in business_validation.items():
            if isinstance(validation, dict) and validation.get("validation") == "FAIL":
                score -= 5  # 5 points for business logic violations

        return max(0, score)

    def _print_quality_summary(self, report):
        """Print executive summary of data quality"""
        print("\n" + "=" * 60)
        print("📊 DATA QUALITY SUMMARY")
        print("=" * 60)

        score = report["quality_score"]

        print(f"Overall Quality Score: {score:.1f}/100")

        if score >= 90:
            quality_level = "EXCELLENT"
        elif score >= 80:
            quality_level = "GOOD"
        elif score >= 70:
            quality_level = "FAIR"
        else:
            quality_level = "POOR - NEEDS ATTENTION"

        print(f"Quality Level: {quality_level}")

        # Key issues summary
        missing_issues = sum(
            1
            for col, analysis in report["missing_data_analysis"].items()
            if analysis["missing_percentage"] > 0
        )
        type_issues = sum(
            1
            for col, analysis in report["data_type_analysis"].items()
            if analysis["conversion_needed"]
        )

        print(f"\nKey Issues Identified:")
        print(f"  • Columns with missing data: {missing_issues}")
        print(f"  • Type conversion needed: {type_issues}")
        print(f"  • Memory usage: {report['dataset_info']['memory_usage_mb']:.2f} MB")

        print(f"\n🔄 Preprocessing Requirements:")
        for col, analysis in report["missing_data_analysis"].items():
            if analysis["missing_percentage"] > 0:
                print(f"  • {col}: {analysis['recommended_strategy']}")

        for col, analysis in report["data_type_analysis"].items():
            if analysis["conversion_needed"]:
                print(f"  • {col}: Convert to {analysis['expected_type']}")


if __name__ == "__main__":
    # Run data quality assessment
    quality_assessor = DataQualityAssessment()
    quality_report = quality_assessor.generate_quality_report()
