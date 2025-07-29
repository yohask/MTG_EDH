# ============================================================================
# MTG Deck Analysis & Recommendation Pipeline
# ============================================================================
# This module implements a comprehensive, research-grade pipeline for analyzing
# Magic: The Gathering (MTG) deck data. It is designed for advanced analytics,
# model development, and explainable recommendations, suitable for a master's
# level project in data science or applied machine learning.
#
# Key Capabilities:
#   - Feature engineering and data cleaning for high-dimensional, heterogeneous data
#   - Unsupervised learning: PCA, clustering, and network analysis for latent structure
#   - Supervised learning: XGBoost, Random Forest, CatBoost, and ensemble regression
#   - Model evaluation: effectiveness, efficiency, and stability across multiple splits
#   - Explainability: SHAP and LIME for model interpretation
#   - Recommender systems: collaborative filtering (ALS), content-based, and hybrid blending
#   - Trend, sentiment, and topic analysis for meta-level insights
#   - Robust CLI for reproducible, end-to-end experimentation
#
# Usage:
#   - Run as a CLI tool for full pipeline execution
#   - Import individual functions for modular analysis or extension
#
# See the main() function for the entry point and argument specification.
# ============================================================================


# === Imports and logger setup ===
import sys
import os
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

# Optional: robust file logging for all errors and debug output
try:
    import logging_setup  # noqa: F401
except Exception as e:
    print("[LOGGING SETUP ERROR]", e, file=sys.stderr)
    import traceback
    traceback.print_exc()



# === Collaborative-filtering backends and flags ===
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


# === 1) LATENT FEATURE EXTRACTION (PCA & CLUSTERING) ===
def extract_latent_features(X: np.ndarray, n_components=10):
    """
    Perform dimensionality reduction and unsupervised clustering on feature matrix X.

    This function applies Principal Component Analysis (PCA) to reduce the dimensionality
    of the input feature space, capturing the most salient variance directions. It then
    applies KMeans clustering to the PCA-transformed data to identify latent groupings
    (e.g., deck archetypes or card clusters) in an unsupervised manner.

    Args:
        X (np.ndarray): High-dimensional feature matrix (e.g., deck or card features).
        n_components (int): Number of principal components to retain.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X_pca: PCA-reduced feature matrix (n_samples x n_components)
            - cluster_labels: Cluster assignments for each sample

    Rationale:
        - PCA is used to mitigate the curse of dimensionality and facilitate visualization.
        - KMeans is chosen for its simplicity and interpretability in grouping similar entities.
        - This step is foundational for downstream EDA and for understanding the latent structure
          of the MTG deck/card space.
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    X_pca = PCA(n_components=n_components).fit_transform(X)
    cluster_labels = KMeans(n_clusters=5, random_state=42).fit_predict(X_pca)
    return X_pca, cluster_labels

# === 2) EXPLORATORY DATA ANALYSIS (CORRELATIONS & NETWORKS) ===
def run_eda(df: pd.DataFrame, output_dir: str):
    """
    Perform advanced exploratory data analysis (EDA) on the MTG deck/card dataset.

    This function generates:
      - Correlation heatmap of all numeric features to reveal linear relationships and multicollinearity.
      - Scatter plots for the top-5 most strongly correlated feature pairs, supporting hypothesis generation.
      - Card–card co-occurrence network graph, visualizing the combinatorial structure of deck construction.
    All outputs are saved to the specified directory for reproducibility and reporting.

    Args:
        df (pd.DataFrame): Main deck/card dataframe.
        output_dir (str): Directory to save all EDA plots and figures.

    Rationale:
        - Correlation analysis is essential for feature selection and understanding redundancy.
        - Co-occurrence networks provide insight into card synergies and meta-level deck design patterns.
        - Visual outputs are critical for both technical and non-technical communication in a research context.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Correlation heatmap: reveals relationships between numeric features ---
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

    # --- Top-5 correlated pairs: scatter plots for strongest relationships ---
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

    # --- Card–card co-occurrence graph: network of cards appearing together in decks ---
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

def train_catboost(X_train, y_train, iterations=100, depth=6, learning_rate=0.1):
    """
    Train a CatBoost regressor for deck/card prediction tasks.
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

def train_mlp_autoencoder(X: np.ndarray, epochs=50, batch_size=32):
    """
    Train an MLP autoencoder for dimensionality reduction and feature learning.
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
# Use FunctionTransformer to ensure TfidfVectorizer gets 1D array
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('text', Pipeline([
        ('extract', FunctionTransformer(lambda x: x[text_features], validate=False)),
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,3)))
    ]), [text_features])
])

# 13) FEATURE ENGINEERING
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # WARNING: To avoid feature leakage, only compute deck-level aggregates or engineered features (such as means, counts, or popularity) on the training split, not the full dataset.
    # DO NOT use viewCount, user interaction counts, or any post-hoc popularity/interaction metrics as predictors or engineered features.
    # If you need to compute aggregates, do so only on the training data after splitting.
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
    # Remove any viewCount or user interaction columns if present (to prevent leakage)
    for col in ['viewCount', 'view_count', 'user_interaction_count', 'user_interactions', 'user_count', 'interaction_count']:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

# 14) MODEL TRAINING
def train_xgb(X_train, y_train, params: dict) -> xgb.Booster:
    """
    Train XGBoost regressor for deck/card prediction tasks.
    Args:
        X_train: Training features.
        y_train: Training targets.
        params: XGBoost parameter dict.
    Returns:
        Trained XGBRegressor model.
    """
    model = xgb.XGBRegressor(**params, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def evaluate_regression(model, X_test, y_test) -> dict:
    """
    Evaluate regression model with RMSE, MAE, R2 for performance assessment.
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test targets.
    Returns:
        Dict of metrics.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return {
        'rmse': rmse,
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

# === MODEL EVALUATION: EFFECTIVENESS, EFFICIENCY, STABILITY ===
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from sklearn.model_selection import cross_val_score

def evaluate_classification(model, X_test, y_test, average='macro'):
    """
    Evaluate classification model with accuracy, precision, recall, F1.
    Args:
        model: Trained classifier.
        X_test: Test features.
        y_test: Test labels.
        average: Averaging method for multiclass (default 'macro').
    Returns:
        Dict of metrics.
    """
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_test, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_test, y_pred, average=average, zero_division=0)
    }

def measure_efficiency(model, X, y=None, task='predict', n_runs=3):
    """
    Measure computational time for fit or predict.
    Args:
        model: Model object.
        X: Features.
        y: Labels (for fit).
        task: 'fit' or 'predict'.
        n_runs: Number of runs to average.
    Returns:
        Average time in seconds.
    """
    times = []
    for _ in range(n_runs):
        start = time.time()
        if task == 'fit':
            model.fit(X, y)
        elif task == 'predict':
            model.predict(X)
        else:
            raise ValueError('task must be "fit" or "predict"')
        times.append(time.time() - start)
    return sum(times) / n_runs

def evaluate_stability(model, X, y, scoring='accuracy', cv=5):
    """
    Assess model stability via cross-validation (returns mean and std of score).
    Args:
        model: Model object.
        X: Features.
        y: Labels.
        scoring: Scoring metric (e.g., 'accuracy', 'neg_root_mean_squared_error').
        cv: Number of folds.
    Returns:
        Dict with mean and std of scores.
    """
    scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    return {'cv_mean': float(np.mean(scores)), 'cv_std': float(np.std(scores))}

# 15) SHAP EXPLAINABILITY
def explain_shap(model, X_sample, output_dir: str):
    """
    Generate SHAP summary plots for model to interpret feature importance.
    Args:
        model: Trained model (XGBRegressor).
        X_sample: Feature matrix.
        output_dir: Directory to save plots.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
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

# 18) BLENDING & PREDICTION
def blend_recommendations(cf_model, content_model, user_id, deck_pool, alpha=0.5) -> List[Any]:
    """
    Blend collaborative and content-based recommendations for final deck ranking.
    
    This function combines scores from a collaborative filtering model (matrix factorization-based, e.g., ALS or SVD)
    and a content-based model (NearestNeighbors with cosine similarity) to produce a hybrid recommendation list.
    
    Args:
        cf_model: Trained collaborative filtering model (matrix factorization-based).
        content_model: Trained content-based NearestNeighbors model (cosine similarity).
        user_id: User index (integer).
        deck_pool: List of deck indices to consider for recommendation.
        alpha: Blend weight (float, 0-1) for collaborative vs. content-based scores.
    Returns:
        List of top-10 recommended deck indices (hardcoded k=10).
    
    Notes:
        - Only the top 10 recommendations are returned (k=10).
        - Cosine similarity is used for content-based recommendations.
        - This function does not support user-based or item-based kNN recommenders.
    """
    # Get collaborative filtering (CF) scores for deck_pool using matrix factorization
    # We need to map between main deck IDs and ALS model's internal indices
    if ALS_AVAILABLE:
        # cf_model was trained on a subset of deck_ids (from interactions)
        # Build mapping from ALS item index to deck_id
        als_deck_ids = cf_model.item_factors.shape[0]
        # The ALS model was trained on categorical codes of deck_id from interactions
        # We'll need to pass in a mapping from deck_id to ALS index
        # For this, we require the mapping as an argument or build it in main()
        raise NotImplementedError("blend_recommendations now requires als_deckid_to_mainidx and mainidx_to_alsdeckid mappings. See main().")
    else:
        cf_vec = np.zeros(len(deck_pool))
    # Content-based scores for deck_pool using cosine similarity
    X_features = content_model._fit_X if hasattr(content_model, '_fit_X') else content_model._fit_X
    pool_features = X_features[deck_pool]
    dists, indices = content_model.kneighbors(pool_features)
    # Use the first neighbor distance for each item (itself)
    content_vec = 1 - dists[:, 0]
    # Normalize both score vectors to [0, 1]
    cf_vec = (cf_vec - cf_vec.min()) / (np.ptp(cf_vec) + 1e-8)
    content_vec = (content_vec - content_vec.min()) / (np.ptp(content_vec) + 1e-8)
    # Blend scores
    blended = alpha * cf_vec + (1 - alpha) * content_vec
    # Return top-10 recommendations
    top_idx = np.argsort(blended)[::-1][:10]
    return [deck_pool[i] for i in top_idx]

# 19) EVALUATION METRICS
def evaluate_recs(true_items: List, pred_items: List, k=10) -> dict:
    """
    Evaluate recommendations with precision@k and NDCG@k for ranking quality.

    Args:
        true_items: List of ground-truth items (e.g., hidden items in hide-k protocol).
        pred_items: List of predicted items (recommendation list).
        k: Top-k cutoff (default 10).
    Returns:
        Dict of metrics: {'precision@k': float, 'ndcg@k': float}

    Notes:
        - Only the top-k items in each list are considered.
        - Used for evaluating recommender system ranking quality.
    """
    # WARNING: To avoid feature leakage, only compute deck-level aggregates or engineered features on the training split, not the full dataset.
# === Reproducibility: Set a global random seed for all random operations (recommended) ===
# import random
# random.seed(42)
# np.random.seed(42)
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
    Main CLI driver for the MTG pipeline.

    This function orchestrates the end-to-end workflow for MTG deck analysis and recommendation.
    It is designed for reproducible, research-grade experimentation and reporting. The pipeline includes:
      - Data loading and cleaning
      - Feature engineering and preprocessing
      - Model training (XGBoost, Random Forest, CatBoost, Ensemble)
      - Model evaluation (effectiveness, efficiency, stability) on two splits
      - Explainability (SHAP)
      - Recommendation (collaborative, content-based, blended)
      - Output of all results for downstream analysis

    Args:
        Uses argparse to specify all input/output paths and parameters.

    Rationale:
        - Modular, extensible, and robust for advanced research and production use.
        - All random seeds are set for reproducibility, a key requirement in scientific work.
        - Outputs are written to disk for traceability and further analysis.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--interactions-path', required=True)
    parser.add_argument('--model-out', required=True)
    parser.add_argument('--shap-out', required=True)
    parser.add_argument('--rec-out', required=True)
    parser.add_argument('--test-user', type=int, default=1, help='User index for recommendations')
    args = parser.parse_args()

    # Set global random seeds for reproducibility (critical for scientific results)
    import random
    random.seed(42)
    np.random.seed(42)

    logger.info("Loading data...")
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)
    df = load_data(args.data_path)
    # Ensure info.oracle_text is string for TfidfVectorizer before any processing
    if 'info.oracle_text' not in df.columns:
        df['info.oracle_text'] = ""
    df['info.oracle_text'] = df['info.oracle_text'].fillna("").apply(str)
    df = add_engineered_features(df)

    # --- Feature/target preparation and cleaning ---
    if 'popularity_score' in df.columns:
        # Clean/fix popularity_score: ensure numeric, fill NaN/inf/extreme values with mean
        y = pd.to_numeric(df['popularity_score'], errors='coerce')
        n_before = y.isna().sum() + np.isinf(y).sum() + (np.abs(y) > 1e10).sum()
        y = y.replace([np.inf, -np.inf], np.nan)
        y[np.abs(y) > 1e10] = np.nan
        if y.isna().sum() > 0:
            mean_val = y[~y.isna()].mean() if (~y.isna()).any() else 0.0
            y = y.fillna(mean_val)
        n_after = y.isna().sum() + np.isinf(y).sum() + (np.abs(y) > 1e10).sum()
        if n_before > 0:
            print(f"[WARNING] Fixed {n_before} invalid values in popularity_score (set to mean)")
        if n_after > 0:
            print(f"[ERROR] Still {n_after} invalid values in popularity_score after cleaning!")
        X = df.drop(columns=['popularity_score'])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    # Ensure info.oracle_text is string in X for TfidfVectorizer
    if 'info.oracle_text' in X.columns:
        X['info.oracle_text'] = X['info.oracle_text'].fillna("").apply(str)

    logger.info("Preprocessing features...")
    # Preprocessing pipeline: numeric, categorical, and text features
    X_proc = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)

    # --- Data integrity check: abort if target is invalid ---
    y_train_arr = np.array(y_train)
    n_nan = np.isnan(y_train_arr).sum()
    n_inf = np.isinf(y_train_arr).sum()
    n_large = np.sum(np.abs(y_train_arr) > 1e10)
    if n_nan > 0 or n_inf > 0 or n_large > 0:
        print("[ERROR] y_train contains NaN, inf, or very large values. Aborting.")
        sys.exit(1)

    # --- Model Training and Evaluation ---
    # Two different train/test splits for robustness (variance estimation)
    from sklearn.ensemble import RandomForestRegressor, VotingRegressor
    try:
        from catboost import CatBoostRegressor
        catboost_available = True
    except ImportError:
        catboost_available = False
    import joblib
    import csv
    from sklearn.utils.multiclass import type_of_target

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_proc, y, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_proc, y, test_size=0.2, random_state=99)
    models = {}
    metrics_all = {}

    # --- Train XGBoost, Random Forest, CatBoost, and Ensemble ---
    logger.info("Training XGBoost model...")
    params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
    model_xgb = train_xgb(X_train1, y_train1, params)
    joblib.dump(model_xgb, args.model_out)
    models['XGBoost'] = model_xgb

    logger.info("Training Random Forest model...")
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train1, y_train1)
    models['RandomForest'] = model_rf

    if catboost_available:
        logger.info("Training CatBoost model...")
        model_cb = CatBoostRegressor(silent=True, iterations=100, depth=6, learning_rate=0.1, random_state=42)
        model_cb.fit(X_train1, y_train1)
        models['CatBoost'] = model_cb

    # Blended Ensemble (VotingRegressor): combines all available models for improved generalization
    estimators = [(k, v) for k, v in models.items()]
    if len(estimators) > 1:
        logger.info("Training VotingRegressor ensemble...")
        ensemble = VotingRegressor(estimators=estimators)
        ensemble.fit(X_train1, y_train1)
        models['Ensemble'] = ensemble

    # --- Model Evaluation: Effectiveness, Efficiency, Stability ---
    def collect_metrics(model, X_test, y_test, X_train, y_train, label, split):
        """
        Collect a comprehensive set of evaluation metrics for a given model and data split.

        This includes:
          - Effectiveness: regression/classification metrics (RMSE, MAE, R2, accuracy, etc.)
          - Efficiency: fit time (proxy for computational cost)
          - Stability: cross-validation mean and std (robustness to data variation)

        Args:
            model: Trained model object
            X_test, y_test: Test features/targets
            X_train, y_train: Training features/targets
            label (str): Model name
            split (int): Split index (1 or 2)

        Returns:
            dict: All metrics for this model/split
        """
        m = {}
        y_type = type_of_target(y_train)
        if y_type in ('binary', 'multiclass'):
            eff = evaluate_classification(model, X_test, y_test)
            m.update({f'{label}_split{split}_accuracy': eff.get('accuracy'),
                      f'{label}_split{split}_precision': eff.get('precision'),
                      f'{label}_split{split}_recall': eff.get('recall'),
                      f'{label}_split{split}_f1': eff.get('f1')})
            eff_time = measure_efficiency(model, X_train, y_train, task='fit', n_runs=1)
            stab = evaluate_stability(model, X_train, y_train, scoring='accuracy', cv=3)
            m[f'{label}_split{split}_efficiency_fit_time'] = eff_time
            m[f'{label}_split{split}_stability_cv_mean'] = stab.get('cv_mean')
            m[f'{label}_split{split}_stability_cv_std'] = stab.get('cv_std')
        else:
            reg = evaluate_regression(model, X_test, y_test)
            m.update({f'{label}_split{split}_rmse': reg.get('rmse'),
                      f'{label}_split{split}_mae': reg.get('mae'),
                      f'{label}_split{split}_r2': reg.get('r2')})
            eff_time = measure_efficiency(model, X_train, y_train, task='fit', n_runs=1)
            stab = evaluate_stability(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=3)
            m[f'{label}_split{split}_efficiency_fit_time'] = eff_time
            m[f'{label}_split{split}_stability_cv_mean'] = stab.get('cv_mean')
            m[f'{label}_split{split}_stability_cv_std'] = stab.get('cv_std')
        return m

    # Evaluate all models on both splits for robust comparison
    for label, model in models.items():
        metrics1 = collect_metrics(model, X_test1, y_test1, X_train1, y_train1, label, 1)
        metrics2 = collect_metrics(model, X_test2, y_test2, X_train2, y_train2, label, 2)
        metrics_all.update(metrics1)
        metrics_all.update(metrics2)

    # Save all metrics to CSV for downstream analysis and reporting
    try:
        with open('model_evaluation_metrics.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['metric', 'value'])
            for k, v in metrics_all.items():
                writer.writerow([k, v])
        print("Model evaluation metrics saved to model_evaluation_metrics.csv")
    except Exception as e:
        print("[CSV save error]", e)
        import traceback
        traceback.print_exc()

    # --- SHAP Explainability: interpretable ML for feature importance ---
    logger.info("Explaining with SHAP...")
    explain_shap(model_xgb, X_train1[:500], args.shap_out)
    logger.info("SHAP explainability done.")

    # --- Recommendation System: collaborative, content-based, and blended ---
    logger.info("Loading interactions...")
    if not os.path.exists(args.interactions_path):
        logger.error(f"Interactions file not found: {args.interactions_path}")
        print("[ERROR] Interactions file not found! Exiting.")
        sys.exit(1)
    interactions = pd.read_csv(args.interactions_path)
    required_cols = {'user_id', 'deck_id'}
    if not required_cols.issubset(interactions.columns):
        print(f"[ERROR] Interactions file missing required columns: {required_cols - set(interactions.columns)}")
        print(f"[ERROR] Available columns: {list(interactions.columns)}")
        sys.exit(1)
    logger.info("Training collaborative filtering model...")
    cf_model = train_cf(interactions)
    logger.info("Collaborative filtering model trained.")
    logger.info("Training content-based recommender...")
    content_model = train_content(X_train)
    logger.info("Content-based recommender trained.")
    logger.info("Blending recommendations...")
    test_user = args.test_user

    # --- Deck pool and mapping logic for hybrid recommendations ---
    # Only recommend decks present in both main data and ALS model
    als_deck_id_cats = interactions['deck_id'].astype('category')
    als_deckid_to_code = dict(zip(als_deck_id_cats.cat.categories, range(len(als_deck_id_cats.cat.categories))))
    code_to_als_deckid = dict(enumerate(als_deck_id_cats.cat.categories))
    if 'id' in df.columns:
        main_deck_ids = df['id'].astype(str).unique().tolist()
    elif 'deck_id' in df.columns:
        main_deck_ids = df['deck_id'].astype(str).unique().tolist()
    else:
        raise ValueError("No deck ID column found in main data (expected 'id' or 'deck_id').")
    shared_deck_ids = [d for d in main_deck_ids if d in als_deckid_to_code]
    if not shared_deck_ids:
        print("[ERROR] No overlapping deck IDs between main data and interactions/ALS model.")
        sys.exit(1)
    mainidx_to_alsdeckid = {i: als_deckid_to_code[d] for i, d in enumerate(shared_deck_ids)}
    alsdeckid_to_mainidx = {als_deckid_to_code[d]: i for i, d in enumerate(shared_deck_ids)}
    deck_pool = list(range(len(shared_deck_ids)))

    # --- Score computation and blending ---
    if ALS_AVAILABLE:
        # Compute collaborative and content-based scores, normalize, and blend
        user_idx = cf_model.user_factors.shape[0] - 1 if test_user >= cf_model.user_factors.shape[0] else test_user
        all_scores = cf_model.user_factors[user_idx] @ cf_model.item_factors.T
        als_indices = [als_deckid_to_code[d] for d in shared_deck_ids]
        cf_vec = np.array([all_scores[i] for i in als_indices])
        X_features = content_model._fit_X if hasattr(content_model, '_fit_X') else content_model._fit_X
        pool_features = X_features[deck_pool]
        dists, indices = content_model.kneighbors(pool_features)
        content_vec = 1 - dists[:, 0]
        # Normalize both score vectors to [0, 1] for fair blending
        cf_vec = (cf_vec - cf_vec.min()) / (np.ptp(cf_vec) + 1e-8)
        content_vec = (content_vec - content_vec.min()) / (np.ptp(content_vec) + 1e-8)
        blended = 0.5 * cf_vec + 0.5 * content_vec
        top_idx = np.argsort(blended)[::-1][:10]
        rec_indices = [deck_pool[i] for i in top_idx]
    else:
        # Fallback: content-based only
        X_features = content_model._fit_X if hasattr(content_model, '_fit_X') else content_model._fit_X
        pool_features = X_features[deck_pool]
        dists, indices = content_model.kneighbors(pool_features)
        content_vec = 1 - dists[:, 0]
        content_vec = (content_vec - content_vec.min()) / (np.ptp(content_vec) + 1e-8)
        top_idx = np.argsort(content_vec)[::-1][:10]
        rec_indices = [deck_pool[i] for i in top_idx]

    # --- Map recommended indices back to deck names (before first ' (') ---
    if 'id' in df.columns and 'name' in df.columns:
        deckid_to_name = dict(zip(df['id'].astype(str), df['name'].astype(str)))
    elif 'deck_id' in df.columns and 'name' in df.columns:
        deckid_to_name = dict(zip(df['deck_id'].astype(str), df['name'].astype(str)))
    else:
        raise ValueError("No deck ID and name columns found in main data.")
    rec_deck_names = []
    for i in rec_indices:
        deck_id = shared_deck_ids[i]
        deck_name_full = deckid_to_name.get(deck_id, str(deck_id))
        deck_name = deck_name_full.split(' (')[0]
        rec_deck_names.append(deck_name)

    # --- Write recommendations to output file for downstream use ---
    with open(args.rec_out, 'w') as f:
        json.dump({'user_id': test_user, 'recommendations': rec_deck_names}, f)
    logger.info("Done. Recommendations written.")


# === Add script entry point ===
if __name__ == "__main__":
    main()

