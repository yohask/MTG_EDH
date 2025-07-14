"""
MTG Deck Analysis & Recommendation Pipeline
-------------------------------------------------
This module provides a comprehensive pipeline for analyzing Magic: The Gathering deck data, including feature engineering, clustering, exploratory analysis, model training, explainability, recommendation, and optimization.

Sections:
    1. Embedding-Based Feature Construction
    2. Latent Feature Extraction (PCA & Clustering)
    3. Exploratory Data Analysis (Correlations & Networks)
    4. Unsupervised Clustering of Deck Themes
    5. Ensemble & Neural Models
    6. Hyperparameter Tuning
    7. Model Explainability (LIME, SHAP)
    8. Learning Curves & Stability Analysis
    9. Sentiment & Topic Evolution
    10. CLI Driver & Example Runner
    11. Design Trend Analysis
    12. Mathematical Programming Deep-Dive

Usage:
    Run as a CLI tool or import functions for custom analysis.
    See the main() and run_examples() functions for entry points.
"""

# === All imports and logger setup below ===
import os
import sys
import argparse
import json
import logging
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import sparse
import datetime
from typing import List, Dict, Any


# Optional collaborative-filtering backends and flags
ALS_AVAILABLE = False
SURPRISE_AVAILABLE = False
AlternatingLeastSquares = None
SurpriseReader = None
SurpriseDataset = None
SurpriseSVD = None
try:
    from implicit.als import AlternatingLeastSquares
    ALS_AVAILABLE = True
except ImportError:
    pass
try:
    from surprise import Dataset as SurpriseDataset, SVD as SurpriseSVD, Reader as SurpriseReader
    SURPRISE_AVAILABLE = True
except ImportError:
    pass

# Configure root logger to INFO
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Module-level logger
logger = logging.getLogger(__name__)
logger.info("Imports and logger are ready.")


# 1) LATENT FEATURE EXTRACTION (PCA & CLUSTERING)
def extract_latent_features(X: np.ndarray, n_components=10):
    """
    Reduce dimensionality with PCA and cluster with KMeans.
    Useful for visualizing latent structure and grouping decks/cards.
    """
    """
    Reduce dimensionality with PCA and cluster with KMeans.
    Args:
        X: Feature matrix.
        n_components: Number of PCA components.
    Returns:
        Tuple of (PCA features, cluster labels).
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    X_pca = PCA(n_components=n_components).fit_transform(X)
    cluster_labels = KMeans(n_clusters=5, random_state=42).fit_predict(X_pca)
    return X_pca, cluster_labels

# 2) EXPLORATORY DATA ANALYSIS (CORRELATIONS & NETWORKS)
def run_eda(df: pd.DataFrame, output_dir: str):
    """
    Run exploratory data analysis:
        - Correlation heatmap of numeric features
        - Top-5 correlated feature pairs (scatter plots)
        - Card–card co-occurrence network graph
    Saves all outputs to the specified directory.
    """
    """
    Run exploratory data analysis: correlation heatmap, top pairs, co-occurrence graph.
    Args:
        df: DataFrame with deck/card data.
        output_dir: Directory to save plots.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Correlation heatmap: reveals relationships between numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found for correlation analysis.")
    corr = df[num_cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    # Top-5 correlated pairs: scatter plots for strongest relationships
    corr_pairs = corr.abs().unstack().sort_values(ascending=False)
    seen = set()
    top_pairs = []
    for (a, b), v in corr_pairs.iteritems():
        if a != b and (b, a) not in seen:
            top_pairs.append((a, b))
            seen.add((a, b))
        if len(top_pairs) >= 5:
            break
    for a, b in top_pairs:
        plt.figure()
        sns.regplot(x=df[a], y=df[b])
        plt.title(f'{a} vs {b}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'scatter_{a}_vs_{b}.png'))
        plt.close()
    # Card–card co-occurrence graph: network of cards appearing together in decks
    # Card–card co-occurrence graph
    if 'deck_id' in df.columns and 'card_id' in df.columns:
        deck_cards = df.groupby('deck_id')['card_id'].apply(list)
        G = nx.Graph()
        for cards in deck_cards:
            for i in range(len(cards)):
                for j in range(i+1, len(cards)):
                    a, b = cards[i], cards[j]
                    if G.has_edge(a, b):
                        G[a][b]['weight'] += 1
                    else:
                        G.add_edge(a, b, weight=1)
        plt.figure(figsize=(10,8))
        pos = nx.spring_layout(G, seed=42)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u,v in edges]
        nx.draw(G, pos, with_labels=True, node_size=50, width=weights, edge_color=weights, edge_cmap=plt.cm.Blues)
        plt.title('Card Co-occurrence Graph')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cooccurrence_graph.png'))
        plt.close()

# 3) UNSUPERVISED CLUSTERING OF DECK THEMES
def cluster_deck_themes(df: pd.DataFrame, output_dir: str):
    """
    Cluster deck themes using DBSCAN and visualize with UMAP.
    Reveals natural groupings in deck text and plots clusters for inspection.
    """
    """
    Cluster deck themes using DBSCAN and visualize with UMAP.
    Args:
        df: DataFrame with deck/card data.
        output_dir: Directory to save plots and cluster mapping.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.cluster import DBSCAN
    import umap
    import matplotlib.pyplot as plt
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    texts = df['oracle_text'].fillna("").astype(str).tolist() if 'oracle_text' in df.columns else df['info.oracle_text'].fillna("").astype(str).tolist()
    X_text = CountVectorizer(max_features=3000).fit_transform(texts)
    labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_text.toarray())
    # Save mapping
    pd.DataFrame({'deck_id': df['deck_id'], 'cluster': labels}).to_csv(os.path.join(output_dir, 'deck_clusters.csv'), index=False)
    # UMAP plot
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X_text.toarray())
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_umap[:,0], X_umap[:,1], c=labels, cmap='Spectral', s=10)
    plt.title('Deck Clusters (UMAP)')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deck_clusters.png'))
    plt.close()

# 4) ADDITIONAL ENSEMBLE & NEURAL MODELS
def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Train a Random Forest regressor for deck/card prediction tasks.
    """
    """
    Train a Random Forest regressor.
    Args:
        X_train: Training features.
        y_train: Training targets.
        n_estimators: Number of trees.
        max_depth: Max tree depth.
    Returns:
        Trained model.
    """
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def train_catboost(X_train, y_train, iterations=100, depth=6, learning_rate=0.1):
    """
    Train a CatBoost regressor for deck/card prediction tasks.
    """
    """
    Train a CatBoost regressor.
    Args:
        X_train: Training features.
        y_train: Training targets.
        iterations: Number of boosting rounds.
        depth: Tree depth.
        learning_rate: Learning rate.
    Returns:
        Trained model.
    """
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(silent=True, iterations=iterations, depth=depth, learning_rate=learning_rate)
    model.fit(X_train, y_train)
    return model
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(silent=True)
    model.fit(X_train, y_train)
    return model

def train_mlp_autoencoder(X: np.ndarray, epochs=50, batch_size=32):
    """
    Train an MLP autoencoder for dimensionality reduction and feature learning.
    """
    """
    Train an MLP autoencoder for dimensionality reduction.
    Args:
        X: Feature matrix.
        epochs: Training epochs.
        batch_size: Batch size.
    Returns:
        Tuple of (autoencoder, encoder).
    """
    from keras.layers import Input, Dense
    from keras.models import Model
    inp = Input(shape=(X.shape[1],))
    encoded = Dense(128, activation='relu')(inp)
    encoded = Dense(64, activation='relu')(encoded)
    bottleneck = Dense(32, activation='relu', name='bottleneck')(encoded)
    decoded = Dense(64, activation='relu')(bottleneck)
    decoded = Dense(128, activation='relu')(decoded)
    out = Dense(X.shape[1], activation='linear')(decoded)
    autoencoder = Model(inp, out)
    encoder = Model(inp, bottleneck)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X, X, epochs=50, batch_size=32, verbose=0)
    return autoencoder, encoder

# 5) HYPERPARAMETER TUNING WITH OPTUNA
def tune_hyperparameters(X_train, y_train, model_type: str, n_trials=50) -> dict:
    """
    Tune model hyperparameters using Optuna for XGBoost, RF, CatBoost, or MLP.
    Returns best parameter set for chosen model type.
    """
    """
    Tune model hyperparameters using Optuna.
    Args:
        X_train: Training features.
        y_train: Training targets.
        model_type: Model type string.
        n_trials: Number of Optuna trials.
    Returns:
        Best parameter dict.
    """
    import optuna
    from sklearn.model_selection import cross_val_score, KFold
    def objective(trial):
        if model_type == 'xgboost':
            import xgboost as xgb
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            model = xgb.XGBRegressor(**params)
        elif model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10)
            }
            model = RandomForestRegressor(**params)
        elif model_type == 'catboost':
            from catboost import CatBoostRegressor
            params = {
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'iterations': trial.suggest_int('iterations', 50, 200),
                'silent': True
            }
            model = CatBoostRegressor(**params)
        elif model_type == 'mlp':
            from keras.models import Sequential
            from keras.layers import Dense
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(64, activation='relu'),
                Dense(1, activation='linear')
            ])
            model.compile(optimizer='adam', loss='mse')
            # Use KFold for keras
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in kf.split(X_train):
                model.fit(X_train[train_idx], y_train[train_idx], epochs=20, batch_size=32, verbose=0)
                preds = model.predict(X_train[val_idx]).flatten()
                rmse = np.sqrt(np.mean((y_train[val_idx] - preds) ** 2))
                scores.append(-rmse)
            return np.mean(scores)
        else:
            raise ValueError('Unknown model_type')
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error').mean()
        return score
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    return study.best_params

# 6) LIME EXPLAINABILITY
def explain_with_lime(model, X_sample: np.ndarray, feature_names: list, output_dir: str):
    """
    Generate LIME explanations for model predictions and save HTML files.
    """
    """
    Generate LIME explanations for model predictions.
    Args:
        model: Trained model.
        X_sample: Feature matrix.
        feature_names: List of feature names.
        output_dir: Directory to save explanations.
    """
    import lime.lime_tabular
    import os
    import random
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    explainer = lime.lime_tabular.LimeTabularExplainer(X_sample, feature_names=feature_names, mode='regression')
    idxs = random.sample(range(X_sample.shape[0]), min(5, X_sample.shape[0]))
    for i, idx in enumerate(idxs):
        exp = explainer.explain_instance(X_sample[idx], model.predict)
        exp.save_to_file(os.path.join(output_dir, f'lime_explanation_{i}.html'))

# 7) LEARNING CURVES & STABILITY ANALYSIS
def plot_learning_curves(estimator, X, y, output_dir: str):
    """
    Plot learning curves and RMSE variance for estimator to assess model stability.
    """
    """
    Plot learning curves and RMSE variance for estimator.
    Args:
        estimator: Model object.
        X: Feature matrix.
        y: Target vector.
        output_dir: Directory to save plots.
    """
    from sklearn.model_selection import learning_curve, KFold
    import matplotlib.pyplot as plt
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, scoring='neg_root_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 5))
    train_rmse = -np.mean(train_scores, axis=1)
    test_rmse = -np.mean(test_scores, axis=1)
    plt.figure()
    plt.plot(train_sizes, train_rmse, label='Train RMSE')
    plt.plot(train_sizes, test_rmse, label='Test RMSE')
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.title('Learning Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'))
    plt.close()
    # Variance of RMSE across folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []
    for train_idx, val_idx in kf.split(X):
        estimator.fit(X[train_idx], y[train_idx])
        preds = estimator.predict(X[val_idx])
        rmses.append(np.sqrt(np.mean((y[val_idx] - preds) ** 2)))
    var_rmse = np.var(rmses)
    print(f'Variance of RMSE across folds: {var_rmse:.4f}')

# 8) SENTIMENT & TOPIC EVOLUTION
def analyze_design_corpus(corpus_dir: str, output_dir: str):
    """
    Analyze design corpus for sentiment and topic evolution using LDA and VADER.
    """
    """
    Analyze design corpus for sentiment and topic evolution.
    Args:
        corpus_dir: Directory of text files.
        output_dir: Directory to save outputs.
    """
    import glob
    import spacy
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    import matplotlib.pyplot as plt
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    nlp = spacy.load('en_core_web_sm')
    analyzer = SentimentIntensityAnalyzer()

    files = glob.glob(os.path.join(corpus_dir, '*.txt'))
    docs = []
    dates = []
    sentiments = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fh:
            text = fh.read()
            doc = nlp(text)
            tokens = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
            docs.append(' '.join(tokens))
            # Try to extract date from filename or file metadata
            try:
                date = os.path.basename(f).split('_')[0]
            except Exception:
                date = 'unknown'
            dates.append(date)
            sentiment = analyzer.polarity_scores(text)['compound']
            sentiments.append(sentiment)
    # Sentiment over time plot
    import pandas as pd
    sentiment_df = pd.DataFrame({'date': dates, 'sentiment': sentiments})
    # Try to sort by date if possible
    try:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce')
        sentiment_df = sentiment_df.sort_values('date')
    except Exception:
        pass
    plt.figure()
    plt.plot(sentiment_df['date'], sentiment_df['sentiment'], marker='o')
    plt.title('Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Compound Sentiment')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_timeseries.png'))
    plt.close()
    # Topic modeling with LDA
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)
    words = np.array(vectorizer.get_feature_names_out())
    topic_words = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:10]
        top_words = words[top_indices]
        topic_words.append({'topic': topic_idx, 'top_words': ', '.join(top_words)})
    pd.DataFrame(topic_words).to_csv(os.path.join(output_dir, 'lda_topics.csv'), index=False)
    # No return, just writes files

# 9) DESIGN TREND ANALYSIS
def run_design_trend_analysis(df: pd.DataFrame, output_dir: str):
    """
    Analyze design trends over time (CMC, keyword density, topics) and save plots/CSVs.
    """
    """
    Analyze design trends over time (CMC, keyword density, topics).
    Args:
        df: DataFrame with deck/card data.
        output_dir: Directory to save outputs.
    """
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    import matplotlib.pyplot as plt
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Ensure date is datetime
    df = df.copy()
    df['deck_release_date'] = pd.to_datetime(df['deck_release_date'], errors='coerce')
    df['release_period'] = df['deck_release_date'].dt.to_period('Q')
    # Compute avg_cmc and avg_keyword_density per period
    keywords = ['flying', 'haste', 'draw']
    def keyword_density(text):
        if not isinstance(text, str):
            return 0
        return sum(text.lower().count(kw) for kw in keywords) / (len(text.split()) if isinstance(text, str) and text.split() else 1)
    df['keyword_density'] = df['info.oracle_text'].apply(keyword_density)
    grouped = df.groupby('release_period').agg(avg_cmc=('cmc', 'mean'), avg_keyword_density=('keyword_density', 'mean')).reset_index()
    # Plot time-series
    plt.figure()
    grouped.plot(x='release_period', y='avg_cmc', marker='o', legend=False)
    plt.title('Average CMC Over Time')
    plt.ylabel('Average CMC')
    plt.xlabel('Release Period')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_cmc_timeseries.png'))
    plt.close()
    plt.figure()
    grouped.plot(x='release_period', y='avg_keyword_density', marker='o', legend=False)
    plt.title('Average Keyword Density Over Time')
    plt.ylabel('Avg Keyword Density')
    plt.xlabel('Release Period')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_keyword_density_timeseries.png'))
    plt.close()
    # LDA topic modeling
    texts = df['info.oracle_text'].fillna("").astype(str)
    vectorizer = CountVectorizer(max_features=2000, stop_words='english')
    X_text = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=6, random_state=42)
    lda.fit(X_text)
    words = np.array(vectorizer.get_feature_names_out())
    topic_words = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:10]
        top_words = words[top_indices]
        topic_words.append({'topic': topic_idx, 'top_words': ', '.join(top_words)})
    pd.DataFrame(topic_words).to_csv(os.path.join(output_dir, 'lda_topics.csv'), index=False)
    # Save grouped stats as CSV
    grouped.to_csv(os.path.join(output_dir, 'trend_stats.csv'), index=False)
    # No return, just writes files

# 10) MATHEMATICAL PROGRAMMING DEEP-DIVE
def solve_deck_lp(card_features: pd.DataFrame, constraints: dict) -> dict:
    """
    Solve deck selection as a linear program (LP) using pulp.
    Maximizes weighted sum of features under deck-building constraints.
    """
    """
    Solve deck selection as a linear program (LP).
    Args:
        card_features: DataFrame with card stats.
        constraints: Dict of LP constraints.
    Returns:
        Dict with selected card IDs and objective value.
    """
    # Requires pulp
    try:
        import pulp
    except ImportError:
        raise ImportError('pulp is required for solve_deck_lp')
    # Assume card_features has columns: 'card_id', 'cmc', 'power', 'toughness', 'keyword_count'
    feature_cols = ['cmc', 'power', 'toughness', 'keyword_count']
    feature_weights = np.array(constraints.get('feature_weights', [1,1,1,1]))
    n = len(card_features)
    prob = pulp.LpProblem('DeckSelection', pulp.LpMaximize)
    x = [pulp.LpVariable(f'x_{i}', cat='Binary') for i in range(n)]
    # Objective: maximize weighted sum
    obj = pulp.lpSum([x[i] * np.dot(feature_weights, card_features.iloc[i][feature_cols]) for i in range(n)])
    prob += obj
    # Constraint: exactly 60 cards
    prob += pulp.lpSum(x) == 60
    # Constraint: total cmc <= max_cmc
    prob += pulp.lpSum([x[i] * card_features.iloc[i]['cmc'] for i in range(n)]) <= constraints['max_cmc']
    # Constraint: total keywords >= min_keyword_count
    prob += pulp.lpSum([x[i] * card_features.iloc[i]['keyword_count'] for i in range(n)]) >= constraints['min_keyword_count']
    # Solve
    prob.solve()
    selected = [int(card_features.iloc[i]['card_id']) for i in range(n) if pulp.value(x[i]) > 0.5]
    return {
        'selected_card_ids': selected,
        'objective_value': pulp.value(prob.objective)
    }

# 11) DATA LOADING
def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV data and print summary stats for inspection.
    """
    """
    Load CSV data and print summary.
    Args:
        path: Path to CSV file.
    Returns:
        DataFrame.
    """
    df = pd.read_csv(path)
    logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(df.info())
    print(df.describe(include='all'))
    return df

# 12) PREPROCESSING PIPELINE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
numeric_features = ['power', 'toughness', 'cmc']
categorical_features = ['color_identity']
text_features = 'info.oracle_text'
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('text', TfidfVectorizer(max_features=5000, ngram_range=(1,3)), text_features)
])

# 13) FEATURE ENGINEERING
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features (days since release, keyword count) for downstream analysis.
    """
    """
    Add engineered features (days since release, keyword count).
    Args:
        df: DataFrame with deck/card data.
    Returns:
        DataFrame with new features.
    """
    df = df.copy()
    today = pd.Timestamp.today()
    df['deck_release_date'] = pd.to_datetime(df['deck_release_date'], errors='coerce')
    df['days_since_release'] = (today - df['deck_release_date']).dt.days
    keywords = ['flying', 'haste', 'draw', 'trample', 'deathtouch', 'lifelink', 'vigilance', 'reach', 'first strike', 'double strike','hexproof', 'menace', 'indestructible', 'protection', 'shroud']
    def count_keywords(text):
        if not isinstance(text, str):
            return 0
        return sum(text.lower().count(kw) for kw in keywords)
    # Use the correct column name for oracle text
    df['keyword_count'] = df['info.oracle_text'].apply(count_keywords)
    return df

# 14) MODEL TRAINING
def train_xgb(X_train, y_train, params: dict) -> xgb.Booster:
    """
    Train XGBoost regressor for deck/card prediction tasks.
    """
    """
    Train XGBoost regressor.
    Args:
        X_train: Training features.
        y_train: Training targets.
        params: XGBoost parameter dict.
    Returns:
        Trained XGBoost Booster.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model

def evaluate_regression(model, X_test, y_test) -> dict:
    """
    Evaluate regression model with RMSE, MAE, R2 for performance assessment.
    """
    """
    Evaluate regression model with RMSE, MAE, R2.
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test targets.
    Returns:
        Dict of metrics.
    """
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return {
        'rmse': rmse,
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

# 15) SHAP EXPLAINABILITY
def explain_shap(model, X_sample, output_dir: str):
    """
    Generate SHAP summary plots for model to interpret feature importance.
    """
    """
    Generate SHAP summary plots for model.
    Args:
        model: Trained model.
        X_sample: Feature matrix.
        output_dir: Directory to save plots.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(os.path.join(output_dir, 'shap_summary_bar.png'))
    plt.close()
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.savefig(os.path.join(output_dir, 'shap_summary_beeswarm.png'))
    plt.close()

# 16) COLLABORATIVE FILTERING
def train_cf(interactions: pd.DataFrame) -> object:
    """
    Train collaborative filtering model (ALS or Surprise) for deck recommendations.
    """
    """
    Train collaborative filtering model (ALS or Surprise).
    Args:
        interactions: DataFrame with user-deck interactions.
    Returns:
        Trained CF model.
    """
    if ALS_AVAILABLE:
        user_ids = interactions['user_id'].astype('category').cat.codes
        deck_ids = interactions['deck_id'].astype('category').cat.codes
        ratings = interactions['rating'] if 'rating' in interactions else interactions['view_count']
        matrix = sparse.coo_matrix((ratings, (user_ids, deck_ids)))
        model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20)
        model.fit(matrix.T)
        return model
    elif SURPRISE_AVAILABLE:
        reader = SurpriseReader(rating_scale=(interactions['rating'].min(), interactions['rating'].max()))
        data = SurpriseDataset.load_from_df(interactions[['user_id','deck_id','rating']], reader)
        trainset = data.build_full_trainset()
        model = SurpriseSVD()
        model.fit(trainset)
        return model
    else:
        raise ImportError("No collaborative filtering library available. Install 'implicit' or 'surprise'.")

# 17) CONTENT-BASED RECOMMENDER
def train_content(X_features: np.ndarray, n_neighbors=10) -> NearestNeighbors:
    """
    Train content-based NearestNeighbors recommender for deck recommendations.
    """
    """
    Train content-based NearestNeighbors recommender.
    Args:
        X_features: Feature matrix.
        n_neighbors: Number of neighbors.
    Returns:
        Trained NearestNeighbors model.
    """
    n_samples = X_features.shape[0]
    n_neighbors = min(n_neighbors, n_samples)
    model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    model.fit(X_features)
    return model
    n_samples = X_features.shape[0]
    n_neighbors = min(10, n_samples)
    model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    model.fit(X_features)
    return model

# 18) BLENDING & PREDICTION
def blend_recommendations(cf_model, content_model, user_id, deck_pool, alpha=0.5) -> List[Any]:
    """
    Blend collaborative and content-based recommendations for final deck ranking.
    """
    """
    Blend collaborative and content-based recommendations.
    Args:
        cf_model: Trained CF model.
        content_model: Trained content model.
        user_id: User index.
        deck_pool: List of deck indices.
        alpha: Blend weight.
    Returns:
        List of top recommended deck indices.
    """
    # Get CF scores for deck_pool
    if ALS_AVAILABLE:
        user_idx = cf_model.user_factors.shape[0] - 1 if user_id >= cf_model.user_factors.shape[0] else user_id
        # Get scores for all items
        all_scores = cf_model.user_factors[user_idx] @ cf_model.item_factors.T
        # Map deck_pool indices to item indices (assume deck_pool are indices into the feature matrix)
        cf_vec = np.array([all_scores[i] for i in deck_pool])
    else:
        cf_vec = np.zeros(len(deck_pool))
    # Content scores for deck_pool
    X_features = content_model._fit_X
    pool_features = X_features[deck_pool]
    dists, indices = content_model.kneighbors(pool_features)
    # Use the first neighbor distance for each item (itself)
    content_vec = 1 - dists[:, 0]
    # Normalize
    cf_vec = (cf_vec - cf_vec.min()) / (np.ptp(cf_vec) + 1e-8)
    content_vec = (content_vec - content_vec.min()) / (np.ptp(content_vec) + 1e-8)
    blended = alpha * cf_vec + (1 - alpha) * content_vec
    top_idx = np.argsort(blended)[::-1][:10]
    return [deck_pool[i] for i in top_idx]

# 19) EVALUATION METRICS
def evaluate_recs(true_items: List, pred_items: List, k=10) -> dict:
    """
    Evaluate recommendations with precision@k and NDCG@k for ranking quality.
    """
    """
    Evaluate recommendations with precision@k and NDCG@k.
    Args:
        true_items: List of true items.
        pred_items: List of predicted items.
        k: Top-k cutoff.
    Returns:
        Dict of metrics.
    """
    def precision_at_k(true, pred, k):
        return len(set(true[:k]) & set(pred[:k])) / float(k)
    def dcg_at_k(rel, k):
        return sum([int(rel[i]) / np.log2(i+2) for i in range(min(len(rel), k))])
    def ndcg_at_k(true, pred, k):
        rel = [1 if item in true[:k] else 0 for item in pred[:k]]
        ideal = sorted(rel, reverse=True)
        return dcg_at_k(rel, k) / (dcg_at_k(ideal, k) + 1e-8)
    return {
        'precision@k': precision_at_k(true_items, pred_items, k),
        'ndcg@k': ndcg_at_k(true_items, pred_items, k)
    }

# 20) CLI DRIVER
def main():
    """
    Main CLI driver for MTG pipeline.
    Parses arguments, runs pipeline, saves outputs.
    Entry point for command-line usage.
    """
    """
    Main CLI driver for MTG pipeline.
    Parses arguments, runs pipeline, saves outputs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--interactions-path', required=True)
    parser.add_argument('--model-out', required=True)
    parser.add_argument('--shap-out', required=True)
    parser.add_argument('--rec-out', required=True)
    args = parser.parse_args()

    logger.info("Loading data...")
    df = load_data(args.data_path)
    # Ensure info.oracle_text is string for TfidfVectorizer before any processing
    if 'info.oracle_text' not in df.columns:
        df['info.oracle_text'] = ""
    df['info.oracle_text'] = df['info.oracle_text'].fillna("").apply(str)
    print("info.oracle_text types:")
    print(df['info.oracle_text'].apply(type).value_counts())
    df = add_engineered_features(df)
    y = df['popularity_score'] if 'popularity_score' in df.columns else df.iloc[:, -1]
    X = df.drop(columns=['popularity_score']) if 'popularity_score' in df.columns else df.iloc[:, :-1]
    # Ensure info.oracle_text is string in X for TfidfVectorizer
    if 'info.oracle_text' in X.columns:
        X['info.oracle_text'] = X['info.oracle_text'].fillna("").apply(str)
    logger.info("Preprocessing features...")
    X_proc = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)
    logger.info("Training XGBoost model...")
    params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
    model = train_xgb(X_train, y_train, params)
    model.save_model(args.model_out)
    logger.info("Evaluating model...")
    metrics = evaluate_regression(model, X_test, y_test)
    logger.info(f"Regression metrics: {metrics}")
    logger.info("Explaining with SHAP...")
    explain_shap(model, X_train[:500], args.shap_out)
    logger.info("Loading interactions...")
    interactions = pd.read_csv(args.interactions_path)
    logger.info("Training collaborative filtering model...")
    cf_model = train_cf(interactions)
    logger.info("Training content-based recommender...")
    content_model = train_content(X_train)
    logger.info("Blending recommendations...")
    test_user = 1
    deck_pool = list(range(X_train.shape[0]))
    recs = blend_recommendations(cf_model, content_model, test_user, deck_pool)
    with open(args.rec_out, 'w') as f:
        json.dump({'user_id': test_user, 'recommendations': recs}, f)
    logger.info("Done. Recommendations written.")

