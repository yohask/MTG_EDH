import os
# Create output directory
os.makedirs('report_figures', exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess dataset
df = pd.read_csv(r'C:\Users\Your\Desktop\mtg assignment\data\Cleaned_Precon_Deck_Data.csv', low_memory=False)

# Compute days since release
df['days_since_release'] = (pd.Timestamp('today') - pd.to_datetime(df['info.released_at'], errors='coerce')).dt.days
# Convert relevant columns to numeric early
for col in ['info.cmc', 'info.power', 'info.toughness', 'days_since_release']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Exclude lands for CMC
spell_df = df[~df['info.type_line'].str.contains('Land', na=False)].copy()

# 1) CMC Histogram with High-CMC Outliers
plt.figure(figsize=(10,6))
counts, bins, patches = plt.hist(
    spell_df['info.cmc'].dropna().astype(int),
    bins=range(0, int(spell_df['info.cmc'].max())+2),
    edgecolor='black', color='#4F81BD', alpha=0.7, label='CMC ≤ 7'
)
# Highlight high-CMC (>=8) bars
high_cmc_indices = [i for i, b in enumerate(bins[:-1]) if b >= 8]
plt.bar(
    [bins[i] for i in high_cmc_indices],
    [counts[i] for i in high_cmc_indices],
    alpha=0.8, edgecolor='black', color='#F79646', label='CMC ≥ 8'
)
plt.title('Distribution of Converted Mana Cost (CMC) for Non-Land Spells\n(High-CMC Cards ≥8 Highlighted)')
plt.xlabel('Converted Mana Cost')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('report_figures/histogram_cmc_outliers.png')
plt.clf()

# 2) Cohort-Normalized Popularity by Release Quarter
plt.figure(figsize=(10,6))
df['quarter'] = pd.to_datetime(df['info.released_at'], errors='coerce').dt.to_period('Q')
coh = df.groupby('quarter')['viewCount'].agg(['mean','std']).dropna()
df = df.join(coh, on='quarter', rsuffix='_cohort')
df['pop_z'] = (df['viewCount'] - df['mean']) / df['std']
quarters = coh.index.astype(str)
z_means = [df[df['quarter']==q]['pop_z'].mean() for q in coh.index]
plt.plot(quarters, z_means, marker='o', color='#4F81BD', label='Mean Z-score')
plt.xticks(rotation=45)
plt.title('Average Deck Popularity (Z-score) by Release Quarter')
plt.xlabel('Release Quarter')
plt.ylabel('Mean Popularity (Z-score)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('report_figures/cohort_popularity_zscore.png')
plt.clf()


# 3) Days Since Release vs. ViewCount Scatter
plt.figure(figsize=(10,6))
# Find the 3 highest and 3 lowest viewCount decks (guaranteed no pd.concat)
extreme_indices = list(df['viewCount'].nlargest(3).index) + list(df['viewCount'].nsmallest(3).index)
# Plot all points in light gray
plt.scatter(df['days_since_release'], df['viewCount'], s=20, alpha=0.2, color='#CCCCCC', label='Deck')
# Plot extremes in orange
plt.scatter(df.loc[extreme_indices, 'days_since_release'], df.loc[extreme_indices, 'viewCount'], s=60, alpha=0.9, color='#F79646', label='Extremes')
# Annotate extremes with name before first '('
for idx in extreme_indices:
    name = df.at[idx, 'name']
    if isinstance(name, str):
        display_name = name.split('(')[0].strip()
        plt.annotate(display_name, (df.at[idx, 'days_since_release'], df.at[idx, 'viewCount']),
                     textcoords="offset points", xytext=(0,12), ha='center', fontsize=12, color='#F79646', weight='bold', fontname='DejaVu Sans')
plt.title('Deck Popularity vs. Age of Release')
plt.xlabel('Days Since Release')
plt.ylabel('ViewCount')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('report_figures/days_vs_viewcount.png')
plt.clf()


from sklearn.cluster import KMeans
# 4) Clustered Deck Scatter Plot (K-means on mean CMC per deck)
deck_means = df.groupby('name')['info.cmc'].mean().reset_index()
deck_views = df.groupby('name')['viewCount'].mean().reset_index()
deck_stats = pd.merge(deck_means, deck_views, on='name')
# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
deck_stats['cluster'] = kmeans.fit_predict(deck_stats[['info.cmc']])
# Assign archetype labels based on mean CMC
archetype_labels = ['Aggro', 'Midrange', 'Control/Combo']
cluster_order = deck_stats.groupby('cluster')['info.cmc'].mean().sort_values().index.tolist()
label_map = {cluster: archetype_labels[i] for i, cluster in enumerate(cluster_order)}
deck_stats['archetype'] = deck_stats['cluster'].map(label_map)
colors = {'Aggro': '#4F81BD', 'Midrange': '#F79646', 'Control/Combo': '#9BBB59'}
plt.figure(figsize=(10,6))
for archetype in archetype_labels:
    subset = deck_stats[deck_stats['archetype'] == archetype]
    plt.scatter(subset['info.cmc'], subset['viewCount'], s=30, alpha=0.6, color=colors[archetype], label=archetype)
    # Highlight and annotate outliers in each cluster (highest and lowest viewCount)
for cluster, label in label_map.items():
    cluster_df = deck_stats[deck_stats['cluster'] == cluster]
    # Find highest and lowest viewCount in cluster
    outlier_indices = list(cluster_df['viewCount'].nlargest(1).index) + list(cluster_df['viewCount'].nsmallest(1).index)
    for idx in outlier_indices:
        row = cluster_df.loc[idx]
        display_name = str(row['name']).split('(')[0].strip()
        plt.scatter(row['info.cmc'], row['viewCount'], s=80, color=colors[label], edgecolor='black', marker='o')
        plt.annotate(display_name, (row['info.cmc'], row['viewCount']), textcoords="offset points", xytext=(0,12), ha='center', fontsize=12, color=colors[label], weight='bold')
plt.title('Deck Archetypes by Mean CMC (K-means Clustering)')
plt.xlabel('Mean CMC per Deck')
plt.ylabel('Average ViewCount per Deck')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('report_figures/deck_cmc_kmeans_scatter.png')
plt.clf()
# 5) Most Common Card Names by Type Category (excluding Basic Lands), skipping tokens, using mana cost or color identity
type_categories = [
    'Artifact', 'Instant', 'Creature', 'Sorcery', 'Enchantment', 'Artifact Creature', 'Land'
]
type_results = []
color_map = {'W': 'White', 'B': 'Black', 'U': 'Blue', 'R': 'Red', 'G': 'Green'}
def format_mana(mana):
    if pd.isna(mana):
        return None
    # Remove curly braces and split
    mana = mana.replace('{', '').replace('}', '')
    # Replace color letters
    for k, v in color_map.items():
        mana = mana.replace(k, v)
    # Split into components and join with spaces
    mana_parts = mana.split()
    return ' '.join(mana_parts) if mana else None
def format_color_identity(ci):
    if pd.isna(ci) or ci == '-' or ci == 'N/A':
        return 'N/A'
    # Replace color letters
    for k, v in color_map.items():
        ci = ci.replace(k, v)
    # Split into components and join with spaces
    ci_parts = ci.split()
    return ' '.join(ci_parts)
for t in type_categories:
    if t == 'Land':
        # Exclude Basic Lands
        mask = df['info.type_line'].str.contains('Land', na=False) & ~df['info.type_line'].str.contains('Basic', na=False)
    else:
        mask = df['info.type_line'].str.contains(t, na=False)
    subset = df[mask]
    # Skip tokens
    subset = subset[subset['info.layout'] != 'token']
    # Get most common card names for this type
    top_cards = subset['info.name'].value_counts().head(10)
    for name, count in top_cards.items():
        card_row = subset[subset['info.name'] == name].iloc[0]
        mana_cost = format_mana(card_row.get('info.mana_cost', None))
        color_identity = format_color_identity(card_row.get('info.color_identity', None))
        if mana_cost:
            color_info = mana_cost
        elif color_identity and color_identity != 'N/A':
            color_info = color_identity
        else:
            color_info = 'N/A'
        # If neither, put '-'
        if not mana_cost and (not color_identity or color_identity == 'N/A'):
            color_info = '-'
        type_results.append({'Type Category': t, 'Card Name': name, 'Mana/Color': color_info, 'Count': count})
import pandas as pd
type_df = pd.DataFrame(type_results)
type_df.to_csv('report_figures/most_common_cards_by_type.csv', index=False)