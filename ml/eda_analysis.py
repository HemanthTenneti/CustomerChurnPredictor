import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

# Add parent directory to Python path to import config
sys.path.append(str(Path(__file__).parent.parent))
import config


class ChurnEDA:
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = config.CHURN_DATA_PATH

        print("📊 Loading Customer Churn Dataset...")
        self.df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

        # Ensure output directories exist
        config.create_directories()

    def basic_data_info(self):
        print("\n" + "=" * 60)
        print("📋 BASIC DATASET INFORMATION")
        print("=" * 60)

        print(f"Dataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        print("\n🏷️ Column Names and Types:")
        for i, (col, dtype) in enumerate(zip(self.df.columns, self.df.dtypes), 1):
            print(f"{i:2d}. {col:<20} ({dtype})")

        print("\n🔍 Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            missing_pct = (missing / len(self.df)) * 100
            for col in missing[missing > 0].index:
                print(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
        else:
            print("  No missing values detected")

        return self.df.info()

    def target_variable_analysis(self):
        print("\n" + "=" * 60)
        print("🎯 TARGET VARIABLE ANALYSIS (Churn)")
        print("=" * 60)

        # Class distribution
        churn_counts = self.df["Churn"].value_counts()
        churn_pcts = self.df["Churn"].value_counts(normalize=True) * 100

        print("Class Distribution:")
        for class_val in churn_counts.index:
            count = churn_counts[class_val]
            pct = churn_pcts[class_val]
            print(f"  {class_val}: {count} customers ({pct:.2f}%)")

        # Class imbalance assessment
        majority_class_pct = churn_pcts.max()
        minority_class_pct = churn_pcts.min()
        imbalance_ratio = majority_class_pct / minority_class_pct

        print(f"\nClass Imbalance Analysis:")
        print(f"  Majority class: {majority_class_pct:.2f}%")
        print(f"  Minority class: {minority_class_pct:.2f}%")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > 1.5:
            print("  ⚠️  Dataset is imbalanced - consider sampling strategies")
        else:
            print("  ✅ Dataset is relatively balanced")

        # Baseline accuracy
        baseline_accuracy = majority_class_pct
        print(f"  Baseline accuracy (majority class): {baseline_accuracy:.2f}%")

        return {
            "churn_distribution": churn_counts,
            "baseline_accuracy": baseline_accuracy,
            "imbalance_ratio": imbalance_ratio,
        }

    def categorical_variables_analysis(self):
        print("\n" + "=" * 60)
        print("📊 CATEGORICAL VARIABLES ANALYSIS")
        print("=" * 60)

        # Identify categorical columns (excluding target)
        categorical_cols = []
        for col in self.df.columns:
            if col != "Churn" and (
                self.df[col].dtype == "object" or self.df[col].nunique() <= 10
            ):
                categorical_cols.append(col)

        print(f"Categorical variables identified: {len(categorical_cols)}")

        categorical_summary = {}
        for col in categorical_cols:
            print(f"\n🔸 {col}:")
            value_counts = self.df[col].value_counts()
            unique_count = self.df[col].nunique()

            print(f"  Unique values: {unique_count}")
            print(f"  Value distribution:")

            for value, count in value_counts.head().items():
                pct = (count / len(self.df)) * 100
                print(f"    {value}: {count} ({pct:.1f}%)")

            if len(value_counts) > 5:
                print(f"    ... and {len(value_counts) - 5} more values")

            categorical_summary[col] = {
                "unique_count": unique_count,
                "value_counts": value_counts,
                "most_frequent": value_counts.index[0],
            }

        return categorical_summary

    def numerical_variables_analysis(self):
        print("\n" + "=" * 60)
        print("📈 NUMERICAL VARIABLES ANALYSIS")
        print("=" * 60)

        # Identify numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Handle TotalCharges if it's stored as object (common issue)
        if (
            "TotalCharges" in self.df.columns
            and self.df["TotalCharges"].dtype == "object"
        ):
            print(
                "⚠️  TotalCharges detected as object type - checking for conversion issues"
            )
            non_numeric = self.df[self.df["TotalCharges"].str.strip() == " "]
            if len(non_numeric) > 0:
                print(f"   Found {len(non_numeric)} rows with empty TotalCharges")

        print(f"Numerical variables identified: {len(numerical_cols)}")

        # Generate descriptive statistics
        print(f"\n📊 Descriptive Statistics:")
        desc_stats = self.df[numerical_cols].describe()
        print(desc_stats.round(2))

        # Analyze each numerical variable
        numerical_summary = {}
        for col in numerical_cols:
            print(f"\n🔹 {col}:")

            data = pd.to_numeric(self.df[col], errors="coerce")

            # Basic statistics
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            min_val = data.min()
            max_val = data.max()

            print(f"  Range: {min_val:.2f} to {max_val:.2f}")
            print(f"  Mean: {mean_val:.2f}, Median: {median_val:.2f}")
            print(f"  Standard Deviation: {std_val:.2f}")

            # Skewness analysis
            skewness = data.skew()
            if abs(skewness) > 1:
                skew_type = "highly skewed"
            elif abs(skewness) > 0.5:
                skew_type = "moderately skewed"
            else:
                skew_type = "approximately normal"

            print(f"  Skewness: {skewness:.2f} ({skew_type})")

            # Outlier detection using IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold_low = Q1 - 1.5 * IQR
            outlier_threshold_high = Q3 + 1.5 * IQR

            outliers = data[
                (data < outlier_threshold_low) | (data > outlier_threshold_high)
            ]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(data)) * 100

            print(f"  Potential outliers: {outlier_count} ({outlier_pct:.1f}%)")

            numerical_summary[col] = {
                "mean": mean_val,
                "median": median_val,
                "std": std_val,
                "skewness": skewness,
                "outlier_count": outlier_count,
                "outlier_percentage": outlier_pct,
            }

        return numerical_summary

    def correlation_analysis(self):
        print("\n" + "=" * 60)
        print("🔗 CORRELATION ANALYSIS")
        print("=" * 60)

        # Get numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) < 2:
            print("⚠️  Need at least 2 numerical variables for correlation analysis")
            return None

        # Calculate correlation matrix
        corr_matrix = self.df[numerical_cols].corr()

        print(f"Correlation Matrix ({len(numerical_cols)} variables):")
        print(corr_matrix.round(3))

        # Identify high correlations
        print(f"\n🔍 High Correlations (|r| > 0.7):")
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    var1 = corr_matrix.columns[i]
                    var2 = corr_matrix.columns[j]
                    high_corr_pairs.append((var1, var2, corr_val))
                    print(f"  {var1} ↔ {var2}: r = {corr_val:.3f}")

        if not high_corr_pairs:
            print("  No high correlations detected")
        else:
            print(
                f"  ⚠️  {len(high_corr_pairs)} high correlation pairs found - consider multicollinearity"
            )

        return {"correlation_matrix": corr_matrix, "high_correlations": high_corr_pairs}

    def generate_visualizations(self):
        print("\n" + "=" * 60)
        print("📊 GENERATING EDA VISUALIZATIONS")
        print("=" * 60)

        # Set style for academic plots
        plt.style.use("default")
        sns.set_palette("husl")

        # Create plots directory
        plots_dir = config.PLOTS_DIR / "eda"
        plots_dir.mkdir(exist_ok=True)

        # 1. Target variable distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Count plot
        sns.countplot(data=self.df, x="Churn", ax=axes[0])
        axes[0].set_title("Churn Distribution (Count)")
        axes[0].set_ylabel("Count")

        # Percentage plot
        churn_pcts = self.df["Churn"].value_counts(normalize=True) * 100
        axes[1].pie(churn_pcts.values, labels=churn_pcts.index, autopct="%1.1f%%")
        axes[1].set_title("Churn Distribution (Percentage)")

        plt.tight_layout()
        plt.savefig(plots_dir / "target_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Numerical variables distributions
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) > 0:
            fig, axes = plt.subplots(
                2, len(numerical_cols), figsize=(5 * len(numerical_cols), 8)
            )
            if len(numerical_cols) == 1:
                axes = axes.reshape(-1, 1)

            for i, col in enumerate(numerical_cols):
                # Convert to numeric (handle TotalCharges issue)
                data = pd.to_numeric(self.df[col], errors="coerce")

                # Histogram
                axes[0, i].hist(data.dropna(), bins=30, alpha=0.7, edgecolor="black")
                axes[0, i].set_title(f"{col} - Histogram")
                axes[0, i].set_xlabel(col)
                axes[0, i].set_ylabel("Frequency")

                # Box plot
                axes[1, i].boxplot(data.dropna())
                axes[1, i].set_title(f"{col} - Box Plot")
                axes[1, i].set_ylabel(col)

            plt.tight_layout()
            plt.savefig(
                plots_dir / "numerical_distributions.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        # 3. Correlation heatmap
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) >= 2:
            plt.figure(figsize=(10, 8))

            # Convert to numeric for correlation
            numeric_df = self.df[numerical_cols].apply(pd.to_numeric, errors="coerce")
            corr_matrix = numeric_df.corr()

            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
                fmt=".3f",
                cbar_kws={"shrink": 0.8},
            )
            plt.title("Correlation Matrix - Numerical Variables")
            plt.tight_layout()
            plt.savefig(
                plots_dir / "correlation_heatmap.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        print(f"✅ Visualizations saved to: {plots_dir}")

        return str(plots_dir)

    def run_full_eda(self):
        print("🚀 STARTING COMPREHENSIVE EDA ANALYSIS")
        print("=" * 80)

        # Run all analysis components
        basic_info = self.basic_data_info()
        target_analysis = self.target_variable_analysis()
        categorical_summary = self.categorical_variables_analysis()
        numerical_summary = self.numerical_variables_analysis()
        correlation_results = self.correlation_analysis()
        plots_path = self.generate_visualizations()

        # Generate summary report
        summary = {
            "dataset_shape": self.df.shape,
            "target_analysis": target_analysis,
            "categorical_summary": categorical_summary,
            "numerical_summary": numerical_summary,
            "correlation_results": correlation_results,
            "plots_location": plots_path,
        }

        print("\n" + "=" * 80)
        print("✅ EDA ANALYSIS COMPLETE")
        print("=" * 80)
        print("📋 Summary of findings:")
        print(f"  • Dataset: {self.df.shape[0]} customers, {self.df.shape[1]} features")
        print(
            f"  • Target distribution: {target_analysis['churn_distribution'].to_dict()}"
        )
        print(f"  • Baseline accuracy: {target_analysis['baseline_accuracy']:.1f}%")
        print(f"  • Categorical variables: {len(categorical_summary)}")
        print(f"  • Numerical variables: {len(numerical_summary)}")
        if correlation_results:
            print(
                f"  • High correlations found: {len(correlation_results['high_correlations'])}"
            )
        print(f"  • Visualizations saved to: {plots_path}")

        return summary


if __name__ == "__main__":
    # Run EDA analysis
    eda = ChurnEDA()
    results = eda.run_full_eda()
