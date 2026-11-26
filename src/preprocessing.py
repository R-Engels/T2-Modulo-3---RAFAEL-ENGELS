"""Fase 4: encoding, split estratificado, balanceamento seguro e escalonamento só para KNN."""

from typing import Any
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    BALANCING_METHOD,
    BINARY_DEFAULT_MAP,
    CONTINUOUS_COLS,
    NOMINAL_COLS,
    ORDINAL_GRADE_MAP,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)


def make_sampler(method: str = BALANCING_METHOD, random_state: int = RANDOM_STATE) -> Any:
    """Instancia o balanceador autorizado ('rus' ou 'smote') para uso exclusivo no treino."""
    key = method.lower().strip()
    if key == "rus":
        return RandomUnderSampler(random_state=random_state)
    if key == "smote":
        return SMOTE(random_state=random_state)
    raise ValueError(f"Método de balanceamento inválido: {method!r}. Use 'rus' ou 'smote'.")


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Converte variáveis categóricas usando estratégias adequadas ao tipo semântico.

    - cb_person_default_on_file: codificação binária (N=0, Y=1)
    - loan_grade: codificação ordinal preservando hierarquia de risco (A=0 .. G=6)
    - person_home_ownership e loan_intent: One-Hot Encoding (nominais sem ordem)
    """
    data = df.copy()

    data["cb_person_default_on_file"] = data["cb_person_default_on_file"].map(
        BINARY_DEFAULT_MAP
    )
    if data["cb_person_default_on_file"].isna().any():
        raise ValueError("Categoria inesperada em cb_person_default_on_file.")
    data["cb_person_default_on_file"] = data["cb_person_default_on_file"].astype(int)

    data["loan_grade"] = data["loan_grade"].map(ORDINAL_GRADE_MAP)
    if data["loan_grade"].isna().any():
        raise ValueError("Categoria inesperada em loan_grade.")
    data["loan_grade"] = data["loan_grade"].astype(int)

    data = pd.get_dummies(
        data,
        columns=NOMINAL_COLS,
        drop_first=True,
        dtype=int,
    )
    return data


def scale_continuous_for_knn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    continuous_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Aplica StandardScaler exclusivamente nas colunas contínuas (fit no treino, transform no teste)."""
    cont_cols = list(continuous_cols or CONTINUOUS_COLS)
    scaler = StandardScaler()
    X_train_knn = X_train.copy()
    X_test_knn = X_test.copy()

    scaled_train = pd.DataFrame(
        scaler.fit_transform(X_train[cont_cols].astype(float)),
        columns=cont_cols,
        index=X_train_knn.index,
    )
    scaled_test = pd.DataFrame(
        scaler.transform(X_test[cont_cols].astype(float)),
        columns=cont_cols,
        index=X_test_knn.index,
    )
    for col in cont_cols:
        X_train_knn[col] = scaled_train[col]
        X_test_knn[col] = scaled_test[col]

    return X_train_knn, X_test_knn, scaler


def split_balance_and_scale(
    df_encoded: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    continuous_cols: list[str] | None = None,
    balancing_method: str = BALANCING_METHOD,
) -> dict[str, Any]:
    """Executa split estratificado 80/20, balanceamento estrito no treino e StandardScaler só para KNN."""
    cont_cols = list(continuous_cols or CONTINUOUS_COLS)

    X = df_encoded.drop(columns=[TARGET_COL])
    y = df_encoded[TARGET_COL]

    # 1. Split estratificado preservando a proporção real das classes
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # 2. Regra de Ouro anti-leakage: balanceamento exclusivamente no Treino
    sampler = make_sampler(balancing_method, random_state=random_state)
    X_train_bal, y_train_bal = sampler.fit_resample(X_train_raw, y_train_raw)

    # 3. Escalonamento seguro exclusivamente nas variáveis contínuas para o KNN
    X_train_knn, X_test_knn, scaler = scale_continuous_for_knn(
        X_train_bal, X_test, continuous_cols=cont_cols
    )

    # 4. Árvore de Decisão usa dados SEM escalonamento (cortes monotônicos independentes de escala)
    X_train_tree = X_train_bal.copy()
    X_test_tree = X_test.copy()

    return {
        "balancing_method": balancing_method.lower().strip(),
        "X_train_raw": X_train_raw,
        "y_train_raw": y_train_raw,
        "X_train_bal": X_train_bal,
        "y_train_bal": y_train_bal,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_knn": X_train_knn,
        "X_test_knn": X_test_knn,
        "X_train_tree": X_train_tree,
        "X_test_tree": X_test_tree,
        "scaler": scaler,
        "continuous_cols": cont_cols,
        "feature_names": list(X.columns),
    }
