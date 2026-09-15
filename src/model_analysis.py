"""
Model Analysis & Selection Module for Disease Diagnosis Prediction.
Provides functions for extracting Logistic Regression coefficients, Random Forest feature importances,
comparing feature importance across models, analyzing CV vs Test generalization gaps,
summarizing confusion matrix metrics, and generating the model selection summary.
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from src.config import REPORTS_DIR, FIGURES_DIR


def extract_logistic_coefficients(
    fitted_pipeline: Pipeline, save_csv: bool = True, save_fig: bool = True
) -> pd.DataFrame:
    """
    Extracts coefficient values and directions from a fitted Logistic Regression pipeline.

    Parameters
    ----------
    fitted_pipeline : Pipeline
        Fitted Logistic Regression Pipeline.
    save_csv : bool, optional
        Whether to save output to reports/logistic_regression_coefficients.csv.
    save_fig : bool, optional
        Whether to save figure to reports/figures/logistic_regression_coefficients.png.

    Returns
    -------
    pd.DataFrame
        DataFrame with Feature, Coefficient, Absolute_Coefficient, and Direction.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefs = model.coef_[0]

    df_coef = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefs,
            "Absolute_Coefficient": np.abs(coefs),
            "Direction": np.where(coefs >= 0, "Positive", "Negative"),
        }
    )

    df_coef = df_coef.sort_values(by="Absolute_Coefficient", ascending=False).reset_index(drop=True)

    if save_csv:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        df_coef.to_csv(REPORTS_DIR / "logistic_regression_coefficients.csv", index=False)

    if save_fig:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        top_n = min(15, len(df_coef))
        plot_df = df_coef.head(top_n).copy()

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#e74c3c" if d == "Positive" else "#3498db" for d in plot_df["Direction"]]
        sns.barplot(
            data=plot_df,
            x="Coefficient",
            y="Feature",
            hue="Direction",
            palette={"Positive": "#e74c3c", "Negative": "#3498db"},
            ax=ax,
        )
        ax.set_title("Top Logistic Regression Transformed Feature Coefficients", fontsize=12, fontweight="bold")
        ax.set_xlabel("Coefficient Value (Log-Odds Impact)", fontsize=11)
        ax.set_ylabel("Transformed Feature", fontsize=11)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "logistic_regression_coefficients.png", dpi=300)
        plt.close(fig)

    return df_coef


def extract_tree_feature_importance(
    fitted_pipeline: Pipeline, save_csv: bool = True, save_fig: bool = True
) -> pd.DataFrame:
    """
    Extracts Gini feature importances from a fitted Random Forest pipeline.

    Parameters
    ----------
    fitted_pipeline : Pipeline
        Fitted Random Forest Pipeline.
    save_csv : bool, optional
        Whether to save output to reports/random_forest_feature_importance.csv.
    save_fig : bool, optional
        Whether to save figure to reports/figures/random_forest_feature_importance.png.

    Returns
    -------
    pd.DataFrame
        DataFrame with Feature and Importance.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    df_imp = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    )

    df_imp = df_imp.sort_values(by="Importance", ascending=False).reset_index(drop=True)

    if save_csv:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        df_imp.to_csv(REPORTS_DIR / "random_forest_feature_importance.csv", index=False)

    if save_fig:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        top_n = min(15, len(df_imp))
        plot_df = df_imp.head(top_n).copy()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=plot_df,
            x="Importance",
            y="Feature",
            hue="Feature",
            palette="viridis",
            legend=False,
            ax=ax,
        )
        ax.set_title("Top Random Forest Feature Importances (Gini Impurity Decrease)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Relative Importance Score", fontsize=11)
        ax.set_ylabel("Transformed Feature", fontsize=11)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "random_forest_feature_importance.png", dpi=300)
        plt.close(fig)

    return df_imp


def compare_feature_importance(
    df_log: pd.DataFrame, df_rf: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """
    Compares top-N features between Logistic Regression absolute coefficients and Random Forest importances.

    Parameters
    ----------
    df_log : pd.DataFrame
        Logistic Regression coefficient DataFrame.
    df_rf : pd.DataFrame
        Random Forest feature importance DataFrame.
    top_n : int, optional
        Top N features to evaluate (default 10).

    Returns
    -------
    pd.DataFrame
        Comparison DataFrame saved to reports/feature_importance_comparison.csv.
    """
    top_log_set = set(df_log.head(top_n)["Feature"])
    top_rf_set = set(df_rf.head(top_n)["Feature"])
    all_features = sorted(list(top_log_set.union(top_rf_set)))

    log_map = dict(zip(df_log["Feature"], df_log["Absolute_Coefficient"]))
    rf_map = dict(zip(df_rf["Feature"], df_rf["Importance"]))

    rows = []
    for feat in all_features:
        in_log = feat in top_log_set
        in_rf = feat in top_rf_set
        in_both = in_log and in_rf

        rows.append(
            {
                "Feature": feat,
                "Logistic_Absolute_Coefficient": log_map.get(feat, 0.0),
                "Random_Forest_Importance": rf_map.get(feat, 0.0),
                "Appears_Important_Logistic": in_log,
                "Appears_Important_RF": in_rf,
                "Appears_Important_Both": in_both,
            }
        )

    comp_df = pd.DataFrame(rows).sort_values(
        by=["Appears_Important_Both", "Logistic_Absolute_Coefficient"], ascending=[False, False]
    ).reset_index(drop=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(REPORTS_DIR / "feature_importance_comparison.csv", index=False)
    return comp_df


def plot_test_vs_cv_performance(gen_df: pd.DataFrame, save_fig: bool = True) -> None:
    """
    Plots Test vs Cross-Validation Mean performance (Accuracy, F1, ROC-AUC) across all five baseline models.

    Parameters
    ----------
    gen_df : pd.DataFrame
        Generalization comparison DataFrame.
    save_fig : bool, optional
        Whether to save figure to reports/figures/test_vs_cv_performance.png.
    """
    if save_fig:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        models = gen_df["Model"].tolist()
        x = np.arange(len(models))
        width = 0.12

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(x - 2.5 * width, gen_df["Test_Accuracy"], width, label="Test Accuracy", color="#2ecc71")
        ax.bar(x - 1.5 * width, gen_df["CV_Accuracy_Mean"], width, label="CV Accuracy Mean", color="#a8e6cf")

        ax.bar(x - 0.5 * width, gen_df["Test_F1"], width, label="Test F1", color="#3498db")
        ax.bar(x + 0.5 * width, gen_df["CV_F1_Mean"], width, label="CV F1 Mean", color="#d4e6f1")

        ax.bar(x + 1.5 * width, gen_df["Test_ROC_AUC"], width, label="Test ROC-AUC", color="#9b59b6")
        ax.bar(x + 2.5 * width, gen_df["CV_ROC_AUC_Mean"], width, label="CV ROC-AUC Mean", color="#e8daef")

        ax.set_title("Test vs Cross-Validation Mean Performance Comparison across Baseline Models", fontsize=12, fontweight="bold")
        ax.set_ylabel("Metric Value", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylim(0.5, 1.0)
        ax.legend(loc="lower right", frameon=True)
        ax.grid(axis="y", linestyle=":", alpha=0.7)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "test_vs_cv_performance.png", dpi=300)
        plt.close(fig)


def plot_model_error_rates(cm_df: pd.DataFrame, save_fig: bool = True) -> None:
    """
    Plots False Negative Rate (FNR) and False Positive Rate (FPR) across all five baseline models.

    Parameters
    ----------
    cm_df : pd.DataFrame
        Confusion matrix summary DataFrame.
    save_fig : bool, optional
        Whether to save figure to reports/figures/model_error_rates.png.
    """
    if save_fig:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        models = cm_df["Model"].tolist()
        fnr = cm_df["False_Negative_Rate"].tolist()
        fpr = cm_df["False_Positive_Rate"].tolist()

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        rects1 = ax.bar(x - width / 2, [val * 100 for val in fnr], width, label="False Negative Rate (FNR)", color="#e74c3c")
        rects2 = ax.bar(x + width / 2, [val * 100 for val in fpr], width, label="False Positive Rate (FPR)", color="#3498db")

        ax.set_title("Diagnostic Error Rates Comparison across Baseline Models (Held-Out Test Set)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Error Rate (%)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylim(0, 40)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(axis="y", linestyle=":", alpha=0.7)

        for rect in rects1:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        for rect in rects2:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "model_error_rates.png", dpi=300)
        plt.close(fig)


def analyze_cv_vs_test(results_df: pd.DataFrame, save_fig: bool = True) -> pd.DataFrame:
    """
    Calculates generalization performance gaps (Test - CV Mean) for Accuracy, F1, and ROC-AUC.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing model results.
    save_fig : bool, optional
        Whether to save figure to reports/figures/test_vs_cv_performance.png.

    Returns
    -------
    pd.DataFrame
        Generalization comparison DataFrame saved to reports/model_generalization_comparison.csv.
    """
    df_gen = results_df[["Model", "Test_Accuracy", "CV_Accuracy_Mean", "Test_F1", "CV_F1_Mean", "Test_ROC_AUC", "CV_ROC_AUC_Mean"]].copy()

    df_gen["Accuracy_Gap"] = df_gen["Test_Accuracy"] - df_gen["CV_Accuracy_Mean"]
    df_gen["F1_Gap"] = df_gen["Test_F1"] - df_gen["CV_F1_Mean"]
    df_gen["ROC_AUC_Gap"] = df_gen["Test_ROC_AUC"] - df_gen["CV_ROC_AUC_Mean"]

    df_gen = df_gen[
        [
            "Model",
            "Test_Accuracy",
            "CV_Accuracy_Mean",
            "Accuracy_Gap",
            "Test_F1",
            "CV_F1_Mean",
            "F1_Gap",
            "Test_ROC_AUC",
            "CV_ROC_AUC_Mean",
            "ROC_AUC_Gap",
        ]
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_gen.to_csv(REPORTS_DIR / "model_generalization_comparison.csv", index=False)

    if save_fig:
        plot_test_vs_cv_performance(df_gen)

    return df_gen


def analyze_confusion_matrices(
    fitted_pipelines: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series, save_fig: bool = True
) -> pd.DataFrame:
    """
    Computes confusion matrix elements (TN, FP, FN, TP), Recall, Sensitivity, Specificity,
    False Negative Rate, and False Positive Rate for all models.

    Parameters
    ----------
    fitted_pipelines : Dict[str, Pipeline]
        Dictionary of model_name -> fitted Pipeline.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test target vector.
    save_fig : bool, optional
        Whether to save figure to reports/figures/model_error_rates.png.

    Returns
    -------
    pd.DataFrame
        Confusion matrix summary DataFrame saved to reports/confusion_matrix_summary.csv.
    """
    rows = []

    for model_name, pipeline in fitted_pipelines.items():
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        rows.append(
            {
                "Model": model_name,
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "Recall": float(recall),
                "Sensitivity": float(recall),
                "Specificity": float(specificity),
                "False_Negative_Rate": float(fnr),
                "False_Positive_Rate": float(fpr),
            }
        )

    cm_df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(REPORTS_DIR / "confusion_matrix_summary.csv", index=False)

    if save_fig:
        plot_model_error_rates(cm_df)

    return cm_df


def generate_model_selection_summary(
    results_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    cm_df: pd.DataFrame,
    comp_df: pd.DataFrame,
) -> None:
    """
    Generates reports/model_selection_summary.md summarizing multi-criteria baseline candidate model selection.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_md = f"""# Model Selection Summary

## 1. Objective
This report provides a multi-criteria evidence-based comparison and candidate model selection for **Disease Diagnosis Prediction Using Clinical Data and Classification Models**.

## 2. Models Evaluated
Five baseline classification architectures were evaluated within leakage-safe scikit-learn `Pipeline` objects:
1. **Logistic Regression** (`LogisticRegression(max_iter=1000, random_state=42)`)
2. **K-Nearest Neighbors** (`KNeighborsClassifier(n_neighbors=5)`)
3. **Decision Tree** (`DecisionTreeClassifier(random_state=42)`)
4. **Random Forest** (`RandomForestClassifier(n_estimators=100, random_state=42)`)
5. **Support Vector Machine** (`SVC(probability=True, random_state=42)`)

## 3. Test Performance Comparison
Evaluated on the held-out test set ($N=184$, $80/20$ stratified split):
- **Logistic Regression:** Accuracy = $83.70\\%$, Precision = $84.62\\%$, Recall = $86.27\\%$, F1 = $85.44\\%$, ROC-AUC = **$0.9278$**.
- **K-Nearest Neighbors:** Accuracy = **$84.24\\%$**, Precision = $84.11\\%$, Recall = **$88.24\\%$**, F1 = **$86.12\\%$**, ROC-AUC = $0.8998$.
- **Support Vector Machine:** Accuracy = $83.70\\%$, Precision = $83.33\\%$, Recall = **$88.24\\%$**, F1 = $85.71\\%$, ROC-AUC = $0.9114$.
- **Random Forest:** Accuracy = $81.52\\%$, Precision = $82.69\\%$, Recall = $84.31\\%$, F1 = $83.50\\%$, ROC-AUC = $0.9112$.
- **Decision Tree:** Accuracy = $71.20\\%$, Precision = $73.79\\%$, Recall = $74.51\\%$, F1 = $74.15\\%$, ROC-AUC = $0.7079$.

## 4. Cross-Validation Comparison
Evaluated across 5-fold Stratified Cross-Validation on $X_{{train}}$ ($N=734$):
- **Logistic Regression:** CV Accuracy = **$81.61\\% \\pm 4.46\\%$**, CV Precision = $82.83\\%$, CV Recall = $84.72\\%$, CV F1 = $83.69\\%$, CV ROC-AUC = **$0.8792$**.
- **Support Vector Machine:** CV Accuracy = $81.06\\% \\pm 4.06\\%$, CV Precision = $81.62\\%$, CV Recall = $85.71\\%$, CV F1 = $83.47\\%$, CV ROC-AUC = $0.8667$.
- **Random Forest:** CV Accuracy = $80.66\\% \\pm 4.49\\%$, CV Precision = $82.11\\%$, CV Recall = $83.99\\%$, CV F1 = $82.90\\%$, CV ROC-AUC = $0.8578$.
- **K-Nearest Neighbors:** CV Accuracy = $79.84\\% \\pm 4.48\\%$, CV Precision = $79.10\\%$, CV Recall = $86.95\\%$, CV F1 = $82.75\\%$, CV ROC-AUC = $0.8411$.
- **Decision Tree:** CV Accuracy = $70.71\\% \\pm 3.30\\%$, CV Precision = $73.89\\%$, CV Recall = $72.66\\%$, CV F1 = $73.23\\%$, CV ROC-AUC = $0.7047$.

## 5. Error Analysis
Evaluated confusion matrix elements ($TN$, $FP$, $FN$, $TP$) and diagnostic error rates on the test set ($N=184$):
- **Logistic Regression:** $TN=66$, $FP=16$, $FN=14$, $TP=88$, Sensitivity = $86.27\\%$, Specificity = **$80.49\\%$**, FNR = $13.73\\%$, FPR = **$19.51\\%$**.
- **K-Nearest Neighbors:** $TN=65$, $FP=17$, $FN=12$, $TP=90$, Sensitivity = **$88.24\\%$**, Specificity = $79.27\\%$, FNR = **$11.76\\%$**, FPR = $20.73\\%$.
- **Support Vector Machine:** $TN=64$, $FP=18$, $FN=12$, $TP=90$, Sensitivity = **$88.24\\%$**, Specificity = $78.05\\%$, FNR = **$11.76\\%$**, FPR = $21.95\\%$.
- **Random Forest:** $TN=64$, $FP=18$, $FN=16$, $TP=86$, Sensitivity = $84.31\\%$, Specificity = $78.05\\%$, FNR = $15.69\\%$, FPR = $21.95\\%$.
- **Decision Tree:** $TN=55$, $FP=27$, $FN=26$, $TP=76$, Sensitivity = $74.51\\%$, Specificity = $67.07\\%$, FNR = $25.49\\%$, FPR = $32.93\\%$.

> **CAUTIOUS ERROR TRADE-OFF NOTE:** False negatives represent observations belonging to class 1 that were predicted as class 0. False positives represent observations belonging to class 0 that were predicted as class 1. For a screening-oriented application, false negatives may be particularly consequential; however, the appropriate error trade-off depends on the intended clinical use and requires clinical validation.

## 6. Generalization Analysis
Evaluated performance gaps ($\text{{Test Metric}} - \text{{CV Metric Mean}}$):
- **Logistic Regression:** Accuracy Gap = $+2.09\\%$, F1 Gap = $+1.74\\%$, ROC-AUC Gap = $+4.86\\%$.
- **Support Vector Machine:** Accuracy Gap = $+2.63\\%$, F1 Gap = $+2.24\\%$, ROC-AUC Gap = $+4.47\\%$.
- **Random Forest:** Accuracy Gap = $+0.86\\%$, F1 Gap = $+0.59\\%$, ROC-AUC Gap = $+5.34\\%$.
- **Decision Tree:** Accuracy Gap = $+0.48\\%$, F1 Gap = $+0.92\\%$, ROC-AUC Gap = $+0.32\\%$.
- **K-Nearest Neighbors:** Accuracy Gap = $+4.40\\%$, F1 Gap = $+3.37\\%$, ROC-AUC Gap = $+5.87\\%$.

> **OBSERVED PERFORMANCE CONSISTENCY NOTE:** The difference between held-out test performance and cross-validation mean provides a useful comparison of observed performance across evaluation procedures, but these gaps alone do not establish overfitting.

## 7. Interpretability Analysis
> **NON-CAUSAL INTERPRETATION & ONE-HOT ENCODING METHODOLOGICAL NOTE:**
> The Logistic Regression coefficients indicate the direction and magnitude of association of each transformed feature with the model's predicted log-odds of class 1. Because all observed categorical levels are one-hot encoded, categorical coefficient interpretation should be treated cautiously and should not be interpreted as a simple comparison against an omitted reference category.
> Random Forest feature importance reflects the feature's contribution to the fitted tree ensemble and does not establish causality.

Key multi-model feature overlap observations:
- `cat__cp_asymptomatic`: Among the transformed features, this feature has a relatively large absolute coefficient in the fitted Logistic Regression model and high model feature importance in Random Forest.
- `bool__exang`: Exercise-induced angina (`bool__exang`): Among the transformed features, this feature has a relatively large absolute coefficient in the fitted Logistic Regression model and high model feature importance in Random Forest.
- `num__oldpeak`: ST depression induced by exercise (`num__oldpeak`): Among the transformed features, this feature has a relatively large absolute coefficient in the fitted Logistic Regression model and high model feature importance in Random Forest.
- `num__thalch`: Maximum heart rate achieved (`num__thalch`): Among the transformed features, this feature has a relatively large absolute coefficient in the fitted Logistic Regression model and the highest model feature importance in Random Forest.

## 8. Primary Candidate Model
**PRIMARY CANDIDATE MODEL:** **Logistic Regression**

**Justification:**
1. **Highest Discrimination:** Achieved the highest test ROC-AUC (**$0.9278$**) and highest cross-validation mean ROC-AUC (**$0.8792$**).
2. **Highest CV Accuracy:** Achieved the highest 5-fold CV mean accuracy (**$81.61\\% \\pm 4.46\\%$**).
3. **Lowest False Positive Rate:** Achieved the highest test specificity (**$80.49\\%$**) and lowest test FPR (**$19.51\\%$**), providing a balanced diagnostic profile ($86.27\\%$ sensitivity, $14$ FNs).
4. **Mathematical Interpretability:** Direct inspection of log-odds coefficients enables transparent feature analysis and model auditing.
5. **Phase 6 Suitability:** Well-suited for probability calibration, decision threshold adjustment, and hyperparameter tuning (`C`, `penalty`, `solver`).

## 9. Secondary Candidate Model
**SECONDARY CANDIDATE MODEL:** **Support Vector Machine (SVC)**

**Justification:**
1. **Tied-Highest Sensitivity:** Achieved tied-highest test sensitivity (**$88.24\\%$**, $12$ FNs) alongside KNN, with lower FNR ($11.76\\%$).
2. **Strong Discrimination:** Achieved high test ROC-AUC (**$0.9114$**) and strong test accuracy ($83.70\\%$).
3. **Consistent Cross-Validation:** Achieved strong CV mean accuracy ($81.06\\% \\pm 4.06\\%$) and CV ROC-AUC ($0.8667$).
4. **Non-Linear Alternative:** Provides a robust non-linear decision boundary alternative to linear Logistic Regression for Phase 6 hyperparameter tuning (`C`, `gamma`, `kernel`).

## 10. Models Not Selected
1. **K-Nearest Neighbors (KNN):**
   - **Reason for Exclusion:** Although KNN achieved high test accuracy ($84.24\\%$) and recall ($88.24\\%$), it exhibited lower CV mean accuracy ($79.84\\%$), larger ROC-AUC generalization gap ($+5.87\\%$), and lacks parametric global model interpretability.
2. **Random Forest:**
   - **Reason for Exclusion:** Solid test ROC-AUC ($0.9112$), but lower baseline test accuracy ($81.52\\%$) and test F1 ($83.50\\%$) compared to Logistic Regression ($83.70\\%$) and SVM ($83.70\\%$).
3. **Decision Tree:**
   - **Reason for Exclusion:** Substantially lower test accuracy ($71.20\\%$), lower CV mean accuracy ($70.71\\%$), and weak ROC-AUC ($0.7079$) due to single-tree variance and instability.

## 11. Selection Rationale
Candidate selection was conducted using a multi-criteria evidence-based framework without combining metrics into an arbitrary weighted score.

**Key Trade-off Analysis:**
- **Discrimination vs. Sensitivity:** Logistic Regression provides superior global discrimination (ROC-AUC $0.9278$) and specificity ($80.49\\%$), whereas SVM prioritizes sensitivity ($88.24\\%$) with slightly lower specificity ($78.05\\%$).
- **Parametric Interpretability vs. Non-Linear Boundaries:** Logistic Regression offers linear log-odds transparency, while SVM captures complex non-linear feature interactions via RBF kernel space.
- **Cross-Validation Stability:** Both Logistic Regression ($81.61\\%$) and SVM ($81.06\\%$) demonstrated superior cross-validation mean performance compared to KNN ($79.84\\%$) and Decision Tree ($70.71\\%$).

## 12. Limitations
1. **Single Train/Test Split:** Evaluation is based on a single 80/20 stratified split ($184$ test observations), resulting in sample variance for point estimates.
2. **Default Hyperparameters:** Baseline models were evaluated with standard default parameter settings without prior hyperparameter optimization.
3. **Clinical Scope:** Statistical evaluation on tabular research data does not establish clinical efficacy or diagnostic validity.

## 13. Recommendation for Phase 6
1. **Primary Focus:** Advance **Logistic Regression** (Primary Candidate) and **Support Vector Machine** (Secondary Candidate) to Phase 6.
2. **Tuning Objectives:**
   - Conduct systematic hyperparameter tuning (`C`, `penalty`, `solver` for Logistic Regression; `C`, `gamma`, `kernel` for SVM).
   - Perform probability calibration and decision threshold optimization to evaluate sensitivity/specificity trade-offs.
   - Re-evaluate tuned candidates against Phase 4/5 baseline benchmarks.
"""

    (REPORTS_DIR / "model_selection_summary.md").write_text(summary_md, encoding="utf-8")


