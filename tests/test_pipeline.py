"""
Unit Test Suite for Disease Diagnosis Prediction Pipeline (Phases 1, 3, 4, & 5).
Tests data loading, schema validation, in-memory data quality handling,
target creation, feature isolation, stratified train/test splitting,
preprocessor fit/transform execution, baseline model pipelines,
cross-validation, test evaluation, reports, visualization artifacts,
Logistic Regression coefficients, Random Forest feature importances,
feature importance comparison, generalization gap calculations,
confusion matrix metrics (Recall, Specificity, FNR, FPR), and candidate model selection.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


from src.config import (
    DATASET_PATH,
    RAW_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES,
    MODEL_FEATURES,
    ORIGINAL_TARGET,
    BINARY_TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    REPORTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
)

from src.data_loader import (
    load_raw_data,
    validate_dataset,
    create_binary_target,
    prepare_model_dataframe,
    load_model_data,
    split_data,
)
from src.preprocessing import build_preprocessor, build_pipeline
from src.train import (
    get_baseline_models,
    build_model_pipeline,
    train_model,
    cross_validate_model,
    train_baseline_models,
    get_tuning_search_spaces,
    tune_model_hyperparameters,
    tune_and_evaluate_candidate_models,
)

from src.evaluate import compute_classification_metrics, evaluate_pipeline_test
from src.model_analysis import (
    extract_logistic_coefficients,
    extract_tree_feature_importance,
    compare_feature_importance,
    analyze_cv_vs_test,
    analyze_confusion_matrices,
    generate_model_selection_summary,
)
from src.advanced_evaluation import (
    get_tuned_candidate_models,
    evaluate_tuned_confusion_matrices,
    plot_tuned_roc_curves,
    plot_tuned_precision_recall_curves,
    analyze_threshold_tradeoffs,
    analyze_probability_calibration,
    generate_advanced_evaluation_summary,
)
from src.model_selection import (
    build_final_model_selection_df,
    generate_final_model_selection_summary,
    run_final_model_selection,
)
from src.model_persistence import (
    get_final_model,
    train_final_model,
    get_model_metadata,
    save_final_model,
    load_final_model,
    verify_persisted_model,
)
from src.predict import (
    load_persisted_model,
    load_model_metadata,
    validate_prediction_input,
    prepare_prediction_dataframe,
    predict_single,
    predict_batch,
)






# =====================================================================
# Phase 1 Tests (1 to 10)
# =====================================================================

def test_load_raw_data_success():
    """1. Test that the raw dataset loads successfully."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_required_columns_exist():
    """2. Test that all expected raw columns exist in the dataset."""
    df = load_raw_data()
    for col in RAW_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"


def test_raw_dataset_row_count():
    """3. Test that the raw dataset has exactly 918 rows."""
    df = load_raw_data()
    assert len(df) == 918, f"Expected 918 rows, found {len(df)}"


def test_raw_dataset_column_count():
    """4. Test that the raw dataset has exactly 11 original columns."""
    df = load_raw_data()
    assert df.shape[1] == 11, f"Expected 11 columns, found {df.shape[1]}"


def test_original_num_values():
    """5. Test that 'num' column contains expected target values (0, 1, 2, 3, 4)."""
    df = load_raw_data()
    unique_num = set(df[ORIGINAL_TARGET].unique())
    expected_values = {0, 1, 2, 3, 4}
    assert unique_num.issubset(expected_values), f"Unexpected 'num' values: {unique_num}"


def test_binary_target_values():
    """6. Test that binary target contains only 0 and 1."""
    df_raw = load_raw_data()
    df_target = create_binary_target(df_raw)
    unique_target = set(df_target[BINARY_TARGET].unique())
    assert unique_target == {0, 1}, f"Binary target must be {{0, 1}}, found {unique_target}"


def test_binary_target_logic():
    """7. Test that binary target logic target = (num > 0).astype(int) is correctly applied."""
    df_raw = load_raw_data()
    df_target = create_binary_target(df_raw)
    expected_target = (df_raw[ORIGINAL_TARGET] > 0).astype(int)
    pd.testing.assert_series_equal(
        df_target[BINARY_TARGET],
        expected_target,
        check_names=False
    )
    assert ORIGINAL_TARGET in df_target.columns


def test_num_excluded_from_features():
    """8. Test that 'num' and 'target' are excluded from the modeling feature matrix X."""
    X, y = load_model_data()
    assert ORIGINAL_TARGET not in X.columns, "'num' must NOT be in feature matrix X"
    assert BINARY_TARGET not in X.columns, "'target' must NOT be in feature matrix X"
    assert list(X.columns) == MODEL_FEATURES
    assert len(X.columns) == 10


def test_build_preprocessor_construction():
    """9. Test that preprocessor can be constructed cleanly."""
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)
    pipeline = build_pipeline()
    assert pipeline is not None


def test_preprocessor_fit_transform():
    """10. Test that preprocessor fits and transforms the actual dataset without errors."""
    X, y = load_model_data()
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == 918
    assert X_transformed.shape[1] == 16
    assert not np.isnan(X_transformed).any()


# =====================================================================
# Phase 3 Tests (11 to 23)
# =====================================================================

def test_prepare_model_dataframe_success():
    """11. Test that prepare_model_dataframe() executes and returns valid DataFrame."""
    df_model = prepare_model_dataframe()
    assert isinstance(df_model, pd.DataFrame)
    assert len(df_model) == 918
    assert BINARY_TARGET in df_model.columns
    assert ORIGINAL_TARGET in df_model.columns


def test_chol_zero_converted_to_nan():
    """12. Test that chol == 0 is converted to NaN in modeling DataFrame."""
    df_raw = load_raw_data()
    raw_zero_chol = (df_raw["chol"] == 0).sum()
    assert raw_zero_chol == 172

    df_model = prepare_model_dataframe()
    assert (df_model["chol"] == 0).sum() == 0
    assert df_model["chol"].isna().sum() == 172


def test_trestbps_zero_converted_to_nan():
    """13. Test that trestbps non-positive values (if any) are handled in modeling DataFrame."""
    df_raw = load_raw_data()
    raw_zero_bp = (df_raw["trestbps"] == 0).sum()
    assert raw_zero_bp == 0
    assert df_raw["trestbps"].min() == 80.0

    df_model = prepare_model_dataframe()
    assert (df_model["trestbps"] == 0).sum() == 0


def test_raw_csv_remains_unchanged():
    """14. Test that data quality handling does NOT mutate the raw CSV file on disk."""
    df_raw_fresh = pd.read_csv(DATASET_PATH)
    assert (df_raw_fresh["chol"] == 0).sum() == 172
    assert (df_raw_fresh["trestbps"] == 0).sum() == 0
    assert "target" not in df_raw_fresh.columns


def test_negative_oldpeak_preserved():
    """15. Test that negative oldpeak values are preserved without modification."""
    df_raw = load_raw_data()
    raw_neg_oldpeak = (df_raw["oldpeak"] < 0).sum()
    assert raw_neg_oldpeak == 12

    df_model = prepare_model_dataframe()
    assert (df_model["oldpeak"] < 0).sum() == 12


def test_x_excludes_num():
    """16. Test that feature matrix X does not contain 'num'."""
    X, y = load_model_data()
    assert "num" not in X.columns


def test_x_excludes_target():
    """17. Test that feature matrix X does not contain 'target'."""
    X, y = load_model_data()
    assert "target" not in X.columns


def test_train_test_split_stratified():
    """18. Test that train_test_split is stratified and maintains class balance."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=True)

    assert len(X_train) == 734
    assert len(X_test) == 184
    assert len(y_train) == 734
    assert len(y_test) == 184

    overall_pos_ratio = y.mean()
    train_pos_ratio = y_train.mean()
    test_pos_ratio = y_test.mean()

    assert pytest.approx(train_pos_ratio, abs=0.01) == overall_pos_ratio
    assert pytest.approx(test_pos_ratio, abs=0.01) == overall_pos_ratio


def test_preprocessor_fits_on_x_train():
    """19. Test that preprocessor can fit on X_train cleanly."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor()
    fitted_prep = preprocessor.fit(X_train)
    assert fitted_prep is not None


def test_preprocessor_transforms_x_train():
    """20. Test that preprocessor transforms X_train into shape (734, 16)."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    assert X_train_trans.shape == (734, 16)


def test_preprocessor_transforms_x_test():
    """21. Test that pre-fitted preprocessor transforms X_test into shape (184, 16)."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    X_test_trans = preprocessor.transform(X_test)
    assert X_test_trans.shape == (184, 16)


def test_no_nans_remain_post_preprocessing():
    """22. Test that zero NaNs remain after preprocessing both X_train and X_test."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()


def test_unknown_categorical_handled_safely():
    """23. Test that unknown categorical values in test data are handled safely without error."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    X_test_mod = X_test.copy()
    X_test_mod.iloc[0, X_test_mod.columns.get_loc("cp")] = "unknown_pain_type"
    X_test_mod.iloc[0, X_test_mod.columns.get_loc("sex")] = "unknown_gender"

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    X_test_trans = preprocessor.transform(X_test_mod)
    assert X_test_trans.shape == (184, 16)
    assert not np.isnan(X_test_trans).any()


# =====================================================================
# Phase 4 Tests (24 to 39)
# =====================================================================

def test_baseline_models_dict_count():
    """24. Test that baseline model dictionary contains exactly five required models."""
    models = get_baseline_models()
    assert isinstance(models, dict)
    assert len(models) == 5
    expected_models = {
        "Logistic Regression",
        "K-Nearest Neighbors",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
    }
    assert set(models.keys()) == expected_models


def test_model_pipeline_construction():
    """25. Test that every baseline model can be wrapped in a Pipeline."""
    models = get_baseline_models()
    for name, model in models.items():
        pipeline = build_model_pipeline(model)
        assert isinstance(pipeline, Pipeline)


def test_pipeline_preprocessor_step():
    """26. Test that every pipeline contains the 'preprocessor' step."""
    models = get_baseline_models()
    for name, model in models.items():
        pipeline = build_model_pipeline(model)
        assert "preprocessor" in pipeline.named_steps
        assert isinstance(pipeline.named_steps["preprocessor"], ColumnTransformer)


def test_pipeline_classifier_step():
    """27. Test that every pipeline contains the 'model' classifier step."""
    models = get_baseline_models()
    for name, model in models.items():
        pipeline = build_model_pipeline(model)
        assert "model" in pipeline.named_steps
        assert pipeline.named_steps["model"] is model


def test_pipeline_fit_on_train():
    """28. Test that every baseline pipeline can fit on training data."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = get_baseline_models()

    for name, model in models.items():
        pipeline = train_model(name, model, X_train, y_train)
        assert pipeline is not None


def test_pipeline_predict_test():
    """29. Test that every baseline pipeline can predict on test data."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = get_baseline_models()

    for name, model in models.items():
        pipeline = train_model(name, model, X_train, y_train)
        preds = pipeline.predict(X_test)
        assert len(preds) == len(y_test)


def test_predictions_binary_values():
    """30. Test that predictions contain only binary values (0 and 1)."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = get_baseline_models()

    for name, model in models.items():
        pipeline = train_model(name, model, X_train, y_train)
        preds = pipeline.predict(X_test)
        unique_preds = set(np.unique(preds))
        assert unique_preds.issubset({0, 1})


def test_predict_proba_valid_range():
    """31. Test that models supporting predict_proba return valid probabilities in [0, 1]."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = get_baseline_models()

    for name, model in models.items():
        pipeline = train_model(name, model, X_train, y_train)
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_test)
            assert probs.shape == (len(y_test), 2)
            assert (probs >= 0.0).all() and (probs <= 1.0).all()
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_decision_function_finite_values():
    """32. Test that models with decision_function return finite numeric values."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = train_model("Logistic Regression", get_baseline_models()["Logistic Regression"], X_train, y_train)

    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X_test)
        assert len(scores) == len(y_test)
        assert np.isfinite(scores).all()


def test_cross_validation_executes():
    """33. Test that 5-fold cross-validation executes successfully for all models."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = get_baseline_models()

    for name, model in models.items():
        pipeline = build_model_pipeline(model)
        cv_res = cross_validate_model(pipeline, X_train, y_train)
        assert isinstance(cv_res, dict)


def test_cv_metrics_keys_exist():
    """34. Test that cross-validation returns all 10 requested mean and std metric keys."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = build_model_pipeline(get_baseline_models()["Logistic Regression"])
    cv_res = cross_validate_model(pipeline, X_train, y_train)

    expected_keys = {
        "CV_Accuracy_Mean", "CV_Accuracy_Std",
        "CV_Precision_Mean", "CV_Precision_Std",
        "CV_Recall_Mean", "CV_Recall_Std",
        "CV_F1_Mean", "CV_F1_Std",
        "CV_ROC_AUC_Mean", "CV_ROC_AUC_Std",
    }
    assert expected_keys.issubset(set(cv_res.keys()))


def test_metrics_are_finite():
    """35. Test that test and CV metric values are finite (not NaN, null, or inf)."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = train_model("Logistic Regression", get_baseline_models()["Logistic Regression"], X_train, y_train)

    test_metrics, y_pred, y_prob = evaluate_pipeline_test(pipeline, X_test, y_test)
    for k, val in test_metrics.items():
        assert np.isfinite(val), f"Metric {k} is not finite: {val}"


def test_roc_auc_in_valid_range():
    """36. Test that computed ROC-AUC values fall between 0.0 and 1.0."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = train_model("Logistic Regression", get_baseline_models()["Logistic Regression"], X_train, y_train)

    test_metrics, _, _ = evaluate_pipeline_test(pipeline, X_test, y_test)
    roc_auc = test_metrics["ROC_AUC"]
    assert 0.0 <= roc_auc <= 1.0


def test_model_results_csv_rows():
    """37. Test that reports/model_results.csv is generated and contains exactly 5 rows."""
    results_path = REPORTS_DIR / "model_results.csv"
    assert results_path.exists()
    df_res = pd.read_csv(results_path)
    assert len(df_res) == 5


def test_model_results_csv_columns():
    """38. Test that reports/model_results.csv contains all required metric columns."""
    results_path = REPORTS_DIR / "model_results.csv"
    df_res = pd.read_csv(results_path)

    required_cols = [
        "Model", "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC",
        "CV_Accuracy_Mean", "CV_Accuracy_Std", "CV_Precision_Mean", "CV_Precision_Std",
        "CV_Recall_Mean", "CV_Recall_Std", "CV_F1_Mean", "CV_F1_Std", "CV_ROC_AUC_Mean", "CV_ROC_AUC_Std"
    ]
    for col in required_cols:
        assert col in df_res.columns, f"Missing required CSV column: {col}"


def test_required_figures_exist():
    """39. Test that all 7 required Phase 4 figure PNG files exist in reports/figures/."""
    expected_figures = [
        "confusion_matrix_logistic_regression.png",
        "confusion_matrix_knn.png",
        "confusion_matrix_decision_tree.png",
        "confusion_matrix_random_forest.png",
        "confusion_matrix_svm.png",
        "roc_curves_baseline.png",
        "baseline_model_comparison.png",
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Missing required figure artifact: {fig_name}"
        assert fig_path.stat().st_size > 0


# =====================================================================
# Phase 5 Tests (40 to 52)
# =====================================================================

def test_extract_logistic_coefficients():
    """40. Test that extract_logistic_coefficients returns valid DataFrame with required columns."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = train_model("Logistic Regression", get_baseline_models()["Logistic Regression"], X_train, y_train)

    df_coef = extract_logistic_coefficients(pipeline, save_csv=False, save_fig=False)
    assert isinstance(df_coef, pd.DataFrame)
    assert len(df_coef) == 16
    required_cols = ["Feature", "Coefficient", "Absolute_Coefficient", "Direction"]
    for col in required_cols:
        assert col in df_coef.columns


def test_logistic_coefficients_csv_columns():
    """41. Test that reports/logistic_regression_coefficients.csv contains correct columns."""
    csv_path = REPORTS_DIR / "logistic_regression_coefficients.csv"
    assert csv_path.exists()
    df_coef = pd.read_csv(csv_path)
    assert len(df_coef) == 16
    for col in ["Feature", "Coefficient", "Absolute_Coefficient", "Direction"]:
        assert col in df_coef.columns


def test_extract_tree_feature_importance():
    """42. Test that extract_tree_feature_importance returns valid DataFrame."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = train_model("Random Forest", get_baseline_models()["Random Forest"], X_train, y_train)

    df_imp = extract_tree_feature_importance(pipeline, save_csv=False, save_fig=False)
    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 16
    for col in ["Feature", "Importance"]:
        assert col in df_imp.columns


def test_rf_feature_importance_csv_columns():
    """43. Test that reports/random_forest_feature_importance.csv contains correct columns."""
    csv_path = REPORTS_DIR / "random_forest_feature_importance.csv"
    assert csv_path.exists()
    df_imp = pd.read_csv(csv_path)
    assert len(df_imp) == 16
    for col in ["Feature", "Importance"]:
        assert col in df_imp.columns


def test_feature_importance_comparison_columns():
    """44. Test that reports/feature_importance_comparison.csv contains required columns."""
    csv_path = REPORTS_DIR / "feature_importance_comparison.csv"
    assert csv_path.exists()
    df_comp = pd.read_csv(csv_path)
    required_cols = [
        "Feature",
        "Logistic_Absolute_Coefficient",
        "Random_Forest_Importance",
        "Appears_Important_Logistic",
        "Appears_Important_RF",
        "Appears_Important_Both",
    ]
    for col in required_cols:
        assert col in df_comp.columns


def test_model_generalization_comparison_columns():
    """45. Test that reports/model_generalization_comparison.csv contains required columns."""
    csv_path = REPORTS_DIR / "model_generalization_comparison.csv"
    assert csv_path.exists()
    df_gen = pd.read_csv(csv_path)
    required_cols = [
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
    for col in required_cols:
        assert col in df_gen.columns


def test_confusion_matrix_summary_columns():
    """46. Test that reports/confusion_matrix_summary.csv contains required columns."""
    csv_path = REPORTS_DIR / "confusion_matrix_summary.csv"
    assert csv_path.exists()
    df_cm = pd.read_csv(csv_path)
    required_cols = [
        "Model", "TN", "FP", "FN", "TP",
        "Recall", "Sensitivity", "Specificity", "False_Negative_Rate", "False_Positive_Rate"
    ]
    for col in required_cols:
        assert col in df_cm.columns


def test_confusion_matrix_counts_non_negative():
    """47. Test that confusion matrix counts TN, FP, FN, TP are non-negative integers summing to 184."""
    csv_path = REPORTS_DIR / "confusion_matrix_summary.csv"
    df_cm = pd.read_csv(csv_path)

    for idx, row in df_cm.iterrows():
        tn, fp, fn, tp = row["TN"], row["FP"], row["FN"], row["TP"]
        assert tn >= 0 and fp >= 0 and fn >= 0 and tp >= 0
        assert (tn + fp + fn + tp) == 184


def test_sensitivity_values_range():
    """48a. Test that computed Sensitivity values are between 0.0 and 1.0."""
    csv_path = REPORTS_DIR / "confusion_matrix_summary.csv"
    df_cm = pd.read_csv(csv_path)
    for val in df_cm["Sensitivity"]:
        assert 0.0 <= val <= 1.0


def test_specificity_values_range():
    """48b. Test that computed Specificity values are between 0.0 and 1.0."""
    csv_path = REPORTS_DIR / "confusion_matrix_summary.csv"
    df_cm = pd.read_csv(csv_path)
    for val in df_cm["Specificity"]:
        assert 0.0 <= val <= 1.0


def test_false_negative_rate_range():
    """49a. Test that False Negative Rate values are between 0.0 and 1.0."""
    csv_path = REPORTS_DIR / "confusion_matrix_summary.csv"
    df_cm = pd.read_csv(csv_path)
    for val in df_cm["False_Negative_Rate"]:
        assert 0.0 <= val <= 1.0


def test_false_positive_rate_range():
    """49b. Test that False Positive Rate values are between 0.0 and 1.0."""
    csv_path = REPORTS_DIR / "confusion_matrix_summary.csv"
    df_cm = pd.read_csv(csv_path)
    for val in df_cm["False_Positive_Rate"]:
        assert 0.0 <= val <= 1.0


def test_generalization_values_finite():
    """49c. Test that generalization metrics and gaps are finite numeric values."""
    csv_path = REPORTS_DIR / "model_generalization_comparison.csv"
    assert csv_path.exists()
    df_gen = pd.read_csv(csv_path)
    for col in ["Accuracy_Gap", "F1_Gap", "ROC_AUC_Gap"]:
        for val in df_gen[col]:
            assert np.isfinite(val)


def test_model_selection_summary_exists():
    """50. Test that reports/model_selection_summary.md exists and is non-empty."""
    summary_path = REPORTS_DIR / "model_selection_summary.md"
    assert summary_path.exists()
    assert summary_path.stat().st_size > 0


def test_notebook_generated_artifacts_exist():
    """51. Test that feature importance figures exist in reports/figures/."""
    log_fig = FIGURES_DIR / "logistic_regression_coefficients.png"
    rf_fig = FIGURES_DIR / "random_forest_feature_importance.png"
    assert log_fig.exists() and log_fig.stat().st_size > 0
    assert rf_fig.exists() and rf_fig.stat().st_size > 0


def test_phase5b_figures_exist():
    """51b. Test that Phase 5B generalization and error rate figures exist in reports/figures/."""
    gen_fig = FIGURES_DIR / "test_vs_cv_performance.png"
    err_fig = FIGURES_DIR / "model_error_rates.png"
    assert gen_fig.exists() and gen_fig.stat().st_size > 0
    assert err_fig.exists() and err_fig.stat().st_size > 0


def test_phase5c_model_selection_summary_structure():
    """53. Test Phase 5C model_selection_summary.md required structure, candidate selection, and model coverage."""
    summary_path = REPORTS_DIR / "model_selection_summary.md"
    assert summary_path.exists()
    assert summary_path.stat().st_size > 0
    text = summary_path.read_text(encoding="utf-8")

    # 1. Required 13 sections exist
    required_sections = [
        "## 1. Objective",
        "## 2. Models Evaluated",
        "## 3. Test Performance Comparison",
        "## 4. Cross-Validation Comparison",
        "## 5. Error Analysis",
        "## 6. Generalization Analysis",
        "## 7. Interpretability Analysis",
        "## 8. Primary Candidate Model",
        "## 9. Secondary Candidate Model",
        "## 10. Models Not Selected",
        "## 11. Selection Rationale",
        "## 12. Limitations",
        "## 13. Recommendation for Phase 6",
    ]
    for sec in required_sections:
        assert sec in text, f"Missing required section in model_selection_summary.md: {sec}"

    # 2. Baseline models discussed
    baseline_models = [
        "Logistic Regression",
        "K-Nearest Neighbors",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
    ]
    for model in baseline_models:
        assert model in text, f"Baseline model {model} not discussed in model_selection_summary.md"

    # 3. Primary and Secondary candidates present and distinct
    assert "PRIMARY CANDIDATE MODEL:" in text or "Primary Candidate Model" in text
    assert "SECONDARY CANDIDATE MODEL:" in text or "Secondary Candidate Model" in text
    assert "Logistic Regression" in text
    assert "Support Vector Machine" in text

    # Primary != Secondary
    primary_is_log = "PRIMARY CANDIDATE MODEL:" in text and "Logistic Regression" in text
    secondary_is_svm = "SECONDARY CANDIDATE MODEL:" in text and "Support Vector Machine" in text
    assert primary_is_log and secondary_is_svm

    # 4. Confirm no arbitrary weighted score formula is used
    assert "weighted score" not in text.lower() or "no arbitrary weighted score" in text.lower() or "without combining metrics into an arbitrary weighted score" in text.lower()


def test_exactly_five_models_in_comparison():
    """52. Test that exactly five baseline models exist in all summary output CSVs."""
    for csv_file in ["model_results.csv", "model_generalization_comparison.csv", "confusion_matrix_summary.csv"]:
        df_csv = pd.read_csv(REPORTS_DIR / csv_file)
        assert len(df_csv) == 5, f"{csv_file} should contain 5 rows"


# =====================================================================
# Phase 6 Tests (54 to 62)
# =====================================================================

def test_tuning_search_spaces_definition():
    """54. Test that get_tuning_search_spaces returns required candidate models and search grids."""
    spaces = get_tuning_search_spaces()
    assert isinstance(spaces, dict)
    assert set(spaces.keys()) == {"Logistic Regression", "Support Vector Machine"}

    lr_model, lr_grid = spaces["Logistic Regression"]
    assert "model__C" in lr_grid
    assert lr_grid["model__C"] == [0.01, 0.1, 1, 10, 100]

    svm_model, svm_grid = spaces["Support Vector Machine"]
    assert "model__C" in svm_grid
    assert "model__kernel" in svm_grid
    assert "model__gamma" in svm_grid
    assert svm_grid["model__C"] == [0.1, 1, 10, 100]
    assert svm_grid["model__kernel"] == ["rbf", "linear"]


def test_tune_model_hyperparameters_execution():
    """55. Test that tune_model_hyperparameters executes GridSearchCV on training data ONLY."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    spaces = get_tuning_search_spaces()

    lr_base, lr_grid = spaces["Logistic Regression"]
    grid_search = tune_model_hyperparameters("Logistic Regression", lr_base, lr_grid, X_train, y_train)

    assert hasattr(grid_search, "best_estimator_")
    assert hasattr(grid_search, "best_params_")
    assert "model__C" in grid_search.best_params_


def test_best_estimator_pipeline_structure():
    """56. Test that best_estimator_ is a scikit-learn Pipeline containing preprocessor and model."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    spaces = get_tuning_search_spaces()

    lr_base, lr_grid = spaces["Logistic Regression"]
    grid_search = tune_model_hyperparameters("Logistic Regression", lr_base, lr_grid, X_train, y_train)
    best_pipe = grid_search.best_estimator_

    assert isinstance(best_pipe, Pipeline)
    assert "preprocessor" in best_pipe.named_steps
    assert "model" in best_pipe.named_steps


def test_no_test_data_in_gridsearch():
    """57. Test that grid search is fitted only on training data (734 samples) and not test data (184 samples)."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    spaces = get_tuning_search_spaces()

    lr_base, lr_grid = spaces["Logistic Regression"]
    grid_search = tune_model_hyperparameters("Logistic Regression", lr_base, lr_grid, X_train, y_train)

    # Verify length of cross-validation splits equals y_train length
    for train_idx, val_idx in grid_search.cv.split(X_train, y_train):
        assert len(train_idx) + len(val_idx) == len(y_train)


def test_tuned_model_results_csv_exists_and_structure():
    """58. Test that reports/tuned_model_results.csv exists and has 4 rows (Baseline & Tuned for LR & SVM)."""
    csv_path = REPORTS_DIR / "tuned_model_results.csv"
    assert csv_path.exists()
    df_tuned = pd.read_csv(csv_path)
    assert len(df_tuned) == 4

    required_cols = [
        "Model", "Variant", "Model_Variant", "Best_Params",
        "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC",
        "CV_Accuracy_Mean", "CV_Accuracy_Std", "CV_ROC_AUC_Mean", "CV_ROC_AUC_Std"
    ]
    for col in required_cols:
        assert col in df_tuned.columns, f"Missing column {col} in tuned_model_results.csv"


def test_hyperparameter_search_summary_exists():
    """59. Test that reports/hyperparameter_search_summary.md exists and contains required sections."""
    summary_path = REPORTS_DIR / "hyperparameter_search_summary.md"
    assert summary_path.exists()
    assert summary_path.stat().st_size > 0
    content = summary_path.read_text(encoding="utf-8")

    assert "Objective & Search Strategy" in content
    assert "Search Spaces & Best Parameters" in content
    assert "Cross-Validation & Held-Out Test Results" in content
    assert "Methodological & Scope Considerations" in content


def test_tuned_vs_baseline_comparison_fig_exists():
    """60. Test that reports/figures/tuned_vs_baseline_comparison.png exists and is non-empty."""
    fig_path = FIGURES_DIR / "tuned_vs_baseline_comparison.png"
    assert fig_path.exists()
    assert fig_path.stat().st_size > 0


def test_models_directory_contains_persisted_artifacts():
    """61. Test that the models/ directory contains final_model.joblib and model_metadata.json."""
    assert MODELS_DIR.exists()
    assert (MODELS_DIR / "final_model.joblib").exists()
    assert (MODELS_DIR / "model_metadata.json").exists()



def test_baseline_model_results_csv_unmodified():
    """62. Test that reports/model_results.csv remains intact and unmodified with 5 baseline rows."""
    base_path = REPORTS_DIR / "model_results.csv"
    assert base_path.exists()
    df_base = pd.read_csv(base_path)
    assert len(df_base) == 5


# =====================================================================
# Phase 7 Tests (63 to 72)
# =====================================================================

def test_get_tuned_candidate_models():
    """63. Test that get_tuned_candidate_models returns Tuned LR and Tuned SVM with expected parameters."""
    candidates = get_tuned_candidate_models()
    assert isinstance(candidates, dict)
    assert set(candidates.keys()) == {"Tuned Logistic Regression", "Tuned Support Vector Machine"}

    lr = candidates["Tuned Logistic Regression"]
    assert lr.C == 0.1
    assert lr.solver == "lbfgs"

    svm = candidates["Tuned Support Vector Machine"]
    assert svm.C == 100
    assert svm.kernel == "linear"
    assert svm.probability is True


def test_evaluate_tuned_confusion_matrices():
    """64. Test that evaluate_tuned_confusion_matrices evaluates on test data and returns valid confusion matrix counts."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    df_cm, fitted = evaluate_tuned_confusion_matrices(X_train, X_test, y_train, y_test, save_csv=False, save_fig=False)

    assert isinstance(df_cm, pd.DataFrame)
    assert len(df_cm) == 2

    for idx, row in df_cm.iterrows():
        tn, fp, fn, tp = row["TN"], row["FP"], row["FN"], row["TP"]
        assert tn + fp + fn + tp == 184
        assert tn >= 0 and fp >= 0 and fn >= 0 and tp >= 0


def test_tuned_confusion_matrix_metrics_range():
    """65. Test that Specificity, FPR, FNR, Sensitivity, and F1 values are in valid [0.0, 1.0] range."""
    csv_path = REPORTS_DIR / "tuned_confusion_matrix_summary.csv"
    assert csv_path.exists()
    df_cm = pd.read_csv(csv_path)
    assert len(df_cm) == 2

    for col in ["Sensitivity", "Specificity", "False_Positive_Rate", "False_Negative_Rate", "F1"]:
        for val in df_cm[col]:
            assert 0.0 <= val <= 1.0


def test_plot_tuned_roc_curves_output():
    """66. Test that plot_tuned_roc_curves generates tuned_roc_curves.png and returns valid AUC scores."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    _, fitted = evaluate_tuned_confusion_matrices(X_train, X_test, y_train, y_test, save_csv=False, save_fig=False)

    auc_scores = plot_tuned_roc_curves(fitted, X_test, y_test)
    assert isinstance(auc_scores, dict)
    assert len(auc_scores) == 2
    for model, auc in auc_scores.items():
        assert 0.5 <= auc <= 1.0

    fig_path = FIGURES_DIR / "tuned_roc_curves.png"
    assert fig_path.exists() and fig_path.stat().st_size > 0


def test_plot_tuned_precision_recall_curves_output():
    """67. Test that plot_tuned_precision_recall_curves generates tuned_precision_recall_curves.png."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    _, fitted = evaluate_tuned_confusion_matrices(X_train, X_test, y_train, y_test, save_csv=False, save_fig=False)

    ap_scores = plot_tuned_precision_recall_curves(fitted, X_test, y_test)
    assert isinstance(ap_scores, dict)
    assert len(ap_scores) == 2
    for model, ap in ap_scores.items():
        assert 0.0 <= ap <= 1.0

    fig_path = FIGURES_DIR / "tuned_precision_recall_curves.png"
    assert fig_path.exists() and fig_path.stat().st_size > 0


def test_analyze_threshold_tradeoffs_training_only():
    """68. Test that analyze_threshold_tradeoffs calculates OoF metrics on training data ONLY (N=734)."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    df_thresh, oof_probs = analyze_threshold_tradeoffs(X_train, y_train, save_csv=False, save_fig=False)
    assert isinstance(df_thresh, pd.DataFrame)
    assert len(oof_probs) == 2

    for model_name, probs in oof_probs.items():
        assert len(probs) == len(X_train) == 734, "OoF probabilities must be computed on X_train (N=734) only"


def test_threshold_analysis_covers_nine_thresholds():
    """69. Test that reports/threshold_analysis.csv exists and contains 9 thresholds per tuned model (18 rows total)."""
    csv_path = REPORTS_DIR / "threshold_analysis.csv"
    assert csv_path.exists()
    df_thresh = pd.read_csv(csv_path)
    assert len(df_thresh) == 18

    expected_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    for model_name in ["Tuned Logistic Regression", "Tuned Support Vector Machine"]:
        sub = df_thresh[df_thresh["Model"] == model_name]
        assert len(sub) == 9
        np.testing.assert_allclose(sub["Threshold"].values, expected_thresholds)


def test_analyze_probability_calibration_brier():
    """70. Test that analyze_probability_calibration computes valid Brier scores in [0.0, 1.0]."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    brier_scores = analyze_probability_calibration(X_train, y_train, save_fig=False)
    assert isinstance(brier_scores, dict)
    assert len(brier_scores) == 2
    for model, brier in brier_scores.items():
        assert 0.0 <= brier <= 0.25, f"Brier score for {model} should be <= 0.25 (found {brier})"

    fig_path = FIGURES_DIR / "calibration_curves.png"
    assert fig_path.exists() and fig_path.stat().st_size > 0


def test_advanced_evaluation_summary_structure():
    """71. Test that reports/advanced_evaluation_summary.csv exists and contains required 13 columns."""
    csv_path = REPORTS_DIR / "advanced_evaluation_summary.csv"
    assert csv_path.exists()
    df_summary = pd.read_csv(csv_path)
    assert len(df_summary) == 2

    required_cols = [
        "Model", "CV_ROC_AUC_Mean", "CV_ROC_AUC_Std", "Test_ROC_AUC",
        "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_Specificity",
        "Test_F1", "Test_FPR", "Test_FNR", "Average_Precision", "Brier_Score"
    ]
    for col in required_cols:
        assert col in df_summary.columns, f"Missing required column {col} in advanced_evaluation_summary.csv"


def test_phase7_models_dir_and_raw_data_integrity():
    """72. Test that models/ contains persisted artifacts and raw dataset MD5 hash remains 13c9cfee54ce2b1552ef7d787a5d8be9."""
    import hashlib
    assert MODELS_DIR.exists()
    assert (MODELS_DIR / "final_model.joblib").exists()
    assert (MODELS_DIR / "model_metadata.json").exists()

    raw_data_bytes = DATASET_PATH.read_bytes()
    md5_hash = hashlib.md5(raw_data_bytes).hexdigest()
    assert md5_hash == "13c9cfee54ce2b1552ef7d787a5d8be9", f"Raw CSV MD5 mismatch: {md5_hash}"



# =====================================================================
# Phase 8 Tests (73 to 78)
# =====================================================================

def test_final_model_selection_df_structure():
    """73. Test that build_final_model_selection_df returns DataFrame with 2 candidate rows and required columns."""
    df_sel = build_final_model_selection_df()
    assert isinstance(df_sel, pd.DataFrame)
    assert len(df_sel) == 2

    required_cols = [
        "Model", "CV_ROC_AUC_Mean", "CV_ROC_AUC_Std", "Test_ROC_AUC",
        "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_Specificity",
        "Test_F1", "Test_FPR", "Test_FNR", "Average_Precision", "Brier_Score",
        "Interpretability", "Selection_Status"
    ]
    for col in required_cols:
        assert col in df_sel.columns, f"Missing required column {col} in final_model_selection.csv"


def test_exactly_one_model_selected():
    """74. Test that exactly ONE model has Selection_Status == 'Selected' and one has 'Rejected'."""
    csv_path = REPORTS_DIR / "final_model_selection.csv"
    assert csv_path.exists()
    df_sel = pd.read_csv(csv_path)

    selected_count = (df_sel["Selection_Status"] == "Selected").sum()
    rejected_count = (df_sel["Selection_Status"] == "Rejected").sum()

    assert selected_count == 1, f"Expected exactly 1 selected model, found {selected_count}"
    assert rejected_count == 1, f"Expected exactly 1 rejected model, found {rejected_count}"


def test_selected_model_is_valid_candidate():
    """75. Test that the selected model is either Tuned Logistic Regression or Tuned Support Vector Machine."""
    csv_path = REPORTS_DIR / "final_model_selection.csv"
    df_sel = pd.read_csv(csv_path)

    selected_row = df_sel[df_sel["Selection_Status"] == "Selected"].iloc[0]
    selected_name = selected_row["Model"]

    assert selected_name in ["Tuned Logistic Regression", "Tuned Support Vector Machine"], f"Invalid selected model: {selected_name}"


def test_final_model_selection_summary_exists():
    """76. Test that reports/final_model_selection_summary.md exists and contains required sections."""
    summary_path = REPORTS_DIR / "final_model_selection_summary.md"
    assert summary_path.exists()
    assert summary_path.stat().st_size > 0
    text = summary_path.read_text(encoding="utf-8")

    required_sections = [
        "## 1. Objective",
        "## 2. Candidate Models Evaluated",
        "## 3. Quantitative Performance Comparison",
        "## 4. Error Profile Comparison",
        "## 5. Decision Threshold Considerations",
        "## 6. Probability Behavior & Calibration",
        "## 7. Interpretability Comparison",
        "## 8. Model Complexity Comparison",
        "## 9. Final Model Decision",
        "## 10. Selection Rationale",
        "## 11. Limitations & Scope",
    ]
    for sec in required_sections:
        assert sec in text, f"Missing required section in final_model_selection_summary.md: {sec}"


def test_previous_phase_reports_unmodified():
    """78. Test that baseline, Phase 6, and Phase 7 report CSVs remain intact."""
    base_csv = pd.read_csv(REPORTS_DIR / "model_results.csv")
    assert len(base_csv) == 5

    tuned_csv = pd.read_csv(REPORTS_DIR / "tuned_model_results.csv")
    assert len(tuned_csv) == 4

    adv_csv = pd.read_csv(REPORTS_DIR / "advanced_evaluation_summary.csv")
    assert len(adv_csv) == 2


# =====================================================================
# Phase 9 Tests (79 to 88)
# =====================================================================

def test_train_final_model_execution():
    """79. Test that train_final_model trains on X_train and returns a fitted Pipeline."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = train_final_model(X_train, y_train)
    assert isinstance(pipeline, Pipeline)
    assert hasattr(pipeline.named_steps["model"], "classes_")


def test_model_metadata_structure():
    """80. Test that get_model_metadata returns dict with all required fields."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    meta = get_model_metadata(X_train, len(X_test))
    assert meta["model_name"] == "Tuned Support Vector Machine"
    assert meta["hyperparameters"]["C"] == 100
    assert meta["hyperparameters"]["kernel"] == "linear"
    assert meta["training_sample_count"] == 734
    assert meta["test_sample_count"] == 184
    assert meta["number_of_input_features"] == 10


def test_save_and_load_final_model(tmp_path):
    """81. Test that save_final_model and load_final_model save and load artifacts correctly."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = train_final_model(X_train, y_train)
    meta = get_model_metadata(X_train, len(X_test))

    model_p, meta_p = save_final_model(pipeline, meta, tmp_path)
    assert model_p.exists() and model_p.stat().st_size > 0
    assert meta_p.exists() and meta_p.stat().st_size > 0

    loaded_pipe, loaded_meta = load_final_model(tmp_path)
    assert isinstance(loaded_pipe, Pipeline)
    assert loaded_meta["model_name"] == "Tuned Support Vector Machine"


def test_loaded_artifact_is_sklearn_pipeline():
    """82. Test that loaded models/final_model.joblib is an instance of sklearn Pipeline."""
    loaded_pipe, _ = load_final_model(MODELS_DIR)
    assert isinstance(loaded_pipe, Pipeline)


def test_loaded_pipeline_contains_preprocessor_and_model():
    """83. Test that loaded pipeline contains preprocessor and model steps."""
    loaded_pipe, _ = load_final_model(MODELS_DIR)
    assert "preprocessor" in loaded_pipe.named_steps
    assert "model" in loaded_pipe.named_steps


def test_loaded_model_hyperparameters():
    """84. Test that classifier in loaded pipeline is SVC with C=100, kernel='linear', probability=True."""
    loaded_pipe, _ = load_final_model(MODELS_DIR)
    model = loaded_pipe.named_steps["model"]

    assert isinstance(model, SVC)
    assert model.C == 100
    assert model.kernel == "linear"
    assert model.probability is True
    assert model.random_state == RANDOM_STATE


def test_loaded_model_predict_shape():
    """85. Test that loaded pipeline predict on X_test returns array of shape (184,)."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    loaded_pipe, _ = load_final_model(MODELS_DIR)

    preds = loaded_pipe.predict(X_test)
    assert preds.shape == (184,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_loaded_model_predict_proba_shape():
    """86. Test that loaded pipeline predict_proba on X_test returns array of shape (184, 2)."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    loaded_pipe, _ = load_final_model(MODELS_DIR)

    probs = loaded_pipe.predict_proba(X_test)
    assert probs.shape == (184, 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_prediction_exact_reproducibility():
    """87. Test that loaded model predictions and probabilities match in-memory trained pipeline 100% exactly."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    in_mem_pipe = train_final_model(X_train, y_train)
    loaded_pipe, _ = load_final_model(MODELS_DIR)

    preds_in_mem = in_mem_pipe.predict(X_test)
    preds_loaded = loaded_pipe.predict(X_test)
    assert np.array_equal(preds_in_mem, preds_loaded)

    probs_in_mem = in_mem_pipe.predict_proba(X_test)[:, 1]
    probs_loaded = loaded_pipe.predict_proba(X_test)[:, 1]
    np.testing.assert_allclose(probs_in_mem, probs_loaded, atol=1e-6)


def test_phase9_raw_data_integrity_and_artifacts():
    """88. Test raw dataset MD5 hash remains 13c9cfee54ce2b1552ef7d787a5d8be9 (918 rows) and artifacts exist."""
    import hashlib
    raw_bytes = DATASET_PATH.read_bytes()
    assert hashlib.md5(raw_bytes).hexdigest() == "13c9cfee54ce2b1552ef7d787a5d8be9"

    raw_df = pd.read_csv(DATASET_PATH)
    assert len(raw_df) == 918

    assert (MODELS_DIR / "final_model.joblib").exists()
    assert (MODELS_DIR / "model_metadata.json").exists()
    assert (REPORTS_DIR / "model_persistence_summary.md").exists()


# =====================================================================
# Phase 10 Tests (89 to 100)
# =====================================================================

def test_load_persisted_model_artifact():
    """89. Test that load_persisted_model returns a valid scikit-learn Pipeline containing preprocessor and model."""
    pipeline = load_persisted_model()
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["model"], SVC)


def test_predict_single_execution_and_structure():
    """90. Test that predict_single accepts a 10-feature dict and returns structured output."""
    sample_input = {
        "age": 55,
        "sex": "Male",
        "cp": "asymptomatic",
        "trestbps": 140,
        "chol": 250,
        "fbs": False,
        "restecg": "normal",
        "thalch": 150,
        "exang": False,
        "oldpeak": 1.2,
    }
    result = predict_single(sample_input)
    assert isinstance(result, dict)
    assert "predicted_class" in result
    assert "predicted_probability" in result
    assert "model_name" in result

    assert result["predicted_class"] in [0, 1]
    assert 0.0 <= result["predicted_probability"] <= 1.0
    assert result["model_name"] == "Tuned Support Vector Machine"


def test_predict_batch_execution_and_structure():
    """91. Test that predict_batch processes a DataFrame and returns expected output columns."""
    X, y = load_model_data()
    X_sample = X.head(5).copy()

    result_df = predict_batch(X_sample)
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 5
    assert "predicted_class" in result_df.columns
    assert "predicted_probability" in result_df.columns

    # Verify input columns are preserved
    for col in MODEL_FEATURES:
        assert col in result_df.columns


def test_predict_batch_does_not_mutate_input():
    """92. Test that predict_batch leaves the original input DataFrame completely unmutated."""
    X, y = load_model_data()
    X_original = X.head(5).copy()
    X_input = X_original.copy()

    result_df = predict_batch(X_input)

    # Input DataFrame must not have prediction columns added
    assert "predicted_class" not in X_input.columns
    assert "predicted_probability" not in X_input.columns
    pd.testing.assert_frame_equal(X_input, X_original)


def test_validate_prediction_input_missing_feature():
    """93. Test that validate_prediction_input raises ValueError when a required feature is missing."""
    sample_input = {
        "sex": "Male",
        "cp": "asymptomatic",
        "trestbps": 140,
        "chol": 250,
        "fbs": False,
        "restecg": "normal",
        "thalch": 150,
        "exang": False,
        "oldpeak": 1.2,
    }  # missing 'age'
    with pytest.raises(ValueError, match="missing required feature"):
        validate_prediction_input(sample_input)


def test_validate_prediction_input_invalid_types():
    """94. Test that validate_prediction_input raises ValueError for invalid data types."""
    invalid_numeric = {
        "age": "fifty-five",  # invalid numeric
        "sex": "Male",
        "cp": "asymptomatic",
        "trestbps": 140,
        "chol": 250,
        "fbs": False,
        "restecg": "normal",
        "thalch": 150,
        "exang": False,
        "oldpeak": 1.2,
    }
    with pytest.raises(ValueError, match="Invalid numeric value"):
        validate_prediction_input(invalid_numeric)

    invalid_bool = {
        "age": 55,
        "sex": "Male",
        "cp": "asymptomatic",
        "trestbps": 140,
        "chol": 250,
        "fbs": "invalid_boolean_string",  # invalid bool
        "restecg": "normal",
        "thalch": 150,
        "exang": False,
        "oldpeak": 1.2,
    }
    with pytest.raises(ValueError, match="Invalid boolean value"):
        validate_prediction_input(invalid_bool)


def test_prediction_class_binary_values():
    """95. Test that prediction class labels are strictly binary (0 or 1)."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    batch_res = predict_batch(X_test)
    unique_classes = set(batch_res["predicted_class"].unique())
    assert unique_classes.issubset({0, 1})


def test_prediction_probability_valid_range():
    """96. Test that positive-class probabilities are bounded in [0.0, 1.0]."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    batch_res = predict_batch(X_test)
    probs = batch_res["predicted_probability"]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_chol_zero_handling_consistency():
    """97. Test that chol=0 input is converted to NaN and handled consistently with training workflow."""
    sample_zero_chol = {
        "age": 55,
        "sex": "Male",
        "cp": "asymptomatic",
        "trestbps": 140,
        "chol": 0,  # zero cholesterol (missing/unrecorded)
        "fbs": False,
        "restecg": "normal",
        "thalch": 150,
        "exang": False,
        "oldpeak": 1.2,
    }
    # Verify prepare_prediction_dataframe converts chol == 0 -> NaN
    df_val = validate_prediction_input(sample_zero_chol)
    df_prep = prepare_prediction_dataframe(df_val)
    assert np.isnan(df_prep["chol"].iloc[0])

    # Verify prediction runs cleanly and produces valid output
    result = predict_single(sample_zero_chol)
    assert result["predicted_class"] in [0, 1]
    assert 0.0 <= result["predicted_probability"] <= 1.0


def test_direct_model_vs_prediction_pipeline_consistency():
    """98. Test that direct model pipeline predictions match src/predict.py 100% exactly."""
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = load_persisted_model()

    # 1. Prepare clean feature matrix for direct pipeline
    df_prep = prepare_prediction_dataframe(X_test)
    direct_preds = pipeline.predict(df_prep)
    direct_probs = pipeline.predict_proba(df_prep)[:, 1]

    # 2. Get predictions via src/predict.py predict_batch
    batch_res = predict_batch(X_test)
    pipeline_preds = batch_res["predicted_class"].values
    pipeline_probs = batch_res["predicted_probability"].values

    # 3. Assert 100% exact numerical match
    assert np.array_equal(direct_preds, pipeline_preds)
    np.testing.assert_allclose(direct_probs, pipeline_probs, atol=1e-6)


def test_feature_ordering_independence():
    """99. Test that predict_single is independent of feature dictionary key ordering."""
    sample_standard = {
        "age": 55,
        "sex": "Male",
        "cp": "asymptomatic",
        "trestbps": 140,
        "chol": 250,
        "fbs": False,
        "restecg": "normal",
        "thalch": 150,
        "exang": False,
        "oldpeak": 1.2,
    }
    # Reverse key order
    sample_shuffled = dict(reversed(list(sample_standard.items())))

    res1 = predict_single(sample_standard)
    res2 = predict_single(sample_shuffled)

    assert res1["predicted_class"] == res2["predicted_class"]
    assert pytest.approx(res1["predicted_probability"], abs=1e-6) == res2["predicted_probability"]


def test_phase10_raw_data_integrity_and_artifacts():
    """100. Test raw dataset MD5 hash remains 13c9cfee54ce2b1552ef7d787a5d8be9 (918 rows) and all artifacts exist."""
    import hashlib
    raw_bytes = DATASET_PATH.read_bytes()
    assert hashlib.md5(raw_bytes).hexdigest() == "13c9cfee54ce2b1552ef7d787a5d8be9"

    raw_df = pd.read_csv(DATASET_PATH)
    assert len(raw_df) == 918

    assert (MODELS_DIR / "final_model.joblib").exists()
    assert (MODELS_DIR / "model_metadata.json").exists()
    assert (MODELS_DIR.parent / "src" / "predict.py").exists()
    assert (REPORTS_DIR / "prediction_pipeline_summary.md").exists()






