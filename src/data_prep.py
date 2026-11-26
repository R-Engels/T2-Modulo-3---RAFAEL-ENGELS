"""Fases 1, 2 e 3: carga, sumário EDA, limpeza e engenharia de atributos."""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.config import (
    DATASET_URL,
    FEATURE_ENGINEERED_COL,
    INCOME_CLIP_QUANTILE,
    MAX_VALID_AGE,
    MAX_VALID_EMP_LENGTH,
    RAW_DATA_PATH,
    TARGET_COL,
)


def load_dataset(
    path: Path = RAW_DATA_PATH,
    url: str = DATASET_URL,
) -> pd.DataFrame:
    """Carrega o CSV bruto; baixa do Google Drive se ausente localmente."""
    csv_path = Path(path)
    if not csv_path.exists():
        import gdown

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(url, str(csv_path), quiet=False)
    return pd.read_csv(csv_path)


def eda_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Calcula estatísticas descritivas e diagnósticos iniciais da base."""
    num_df = df.select_dtypes(include="number")
    target_counts = df[TARGET_COL].value_counts().sort_index()
    target_pct = (df[TARGET_COL].value_counts(normalize=True).sort_index() * 100).round(2)

    return {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "describe": df.describe().T,
        "null_counts": df.isna().sum(),
        "null_pct": (df.isna().mean() * 100).round(2),
        "duplicates": int(df.duplicated().sum()),
        "skewness": num_df.skew().round(3),
        "target_counts": target_counts,
        "target_pct": target_pct,
    }


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Executa Fase 2: remoção de duplicadas, tratamento de outliers e imputação de nulos."""
    data = df.copy()
    initial_rows = len(data)

    # 1. Remoção de linhas duplicadas exatas
    data = data.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_rows - len(data)

    # 2. Remoção de anomalias fisicamente impossíveis (erros de registro)
    invalid_age_mask = data["person_age"] > MAX_VALID_AGE
    invalid_emp_mask = data["person_emp_length"] > MAX_VALID_EMP_LENGTH
    invalid_age_count = int(invalid_age_mask.sum())
    invalid_emp_count = int(invalid_emp_mask.sum())
    data = data.loc[~(invalid_age_mask | invalid_emp_mask)].reset_index(drop=True)

    # 3. Clipping (winsorização) de renda extrema no percentil 99 para proteger o KNN
    income_cap = float(data["person_income"].quantile(INCOME_CLIP_QUANTILE))
    income_clipped_count = int((data["person_income"] > income_cap).sum())
    data["person_income"] = data["person_income"].clip(upper=income_cap)

    # 4. Imputação estatística fundamentada de nulos
    # person_emp_length: assimetria forte à direita -> Mediana
    emp_nulls = int(data["person_emp_length"].isna().sum())
    emp_median = float(data["person_emp_length"].median())
    data["person_emp_length"] = data["person_emp_length"].fillna(emp_median)

    # loan_int_rate: distribuição aproximadamente simétrica (|skew| ~ 0.21) -> Média
    rate_nulls = int(data["loan_int_rate"].isna().sum())
    rate_mean = float(data["loan_int_rate"].mean())
    data["loan_int_rate"] = data["loan_int_rate"].fillna(rate_mean)

    report = {
        "initial_rows": initial_rows,
        "duplicates_removed": duplicates_removed,
        "invalid_age_removed": invalid_age_count,
        "invalid_emp_removed": invalid_emp_count,
        "income_cap_p99": income_cap,
        "income_clipped_count": income_clipped_count,
        "emp_length_nulls_imputed": emp_nulls,
        "emp_length_median": emp_median,
        "int_rate_nulls_imputed": rate_nulls,
        "int_rate_mean": round(rate_mean, 4),
        "final_rows": len(data),
        "remaining_nulls": int(data.isna().sum().sum()),
    }
    return data, report


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Executa Fase 3: cria coluna calculada comprometimento_renda sem NaN/inf."""
    data = df.copy()

    if data[["loan_amnt", "person_income"]].isna().any().any():
        raise ValueError("Nulos encontrados em loan_amnt ou person_income antes do cálculo.")
    if (data["person_income"] <= 0).any():
        raise ValueError("Valores não positivos encontrados em person_income (risco de divisão por zero).")

    data[FEATURE_ENGINEERED_COL] = (data["loan_amnt"] / data["person_income"]) * 100.0

    if not np.isfinite(data[FEATURE_ENGINEERED_COL]).all():
        raise ValueError("Coluna comprometimento_renda contém NaN ou infinito.")

    return data
