import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

############################
# 1. LOAD & PREPARE DATA
############################

DATA_PATH = r"C:\Users\zuzia\Downloads\soc-redditHyperlinks-body.tsv"

df_links = pd.read_csv(DATA_PATH, sep='\t')
df_links['TIMESTAMP'] = pd.to_datetime(df_links['TIMESTAMP'])

print("Columns:", df_links.columns.tolist())

# We only care about the date range 2014-01-01 to 2017-06-30 overall,
# so let's do one global filter first (optional but can reduce data size).
start_global = "2014-01-01"
end_global   = "2017-06-30"

df_links = df_links[(df_links['TIMESTAMP'] >= start_global) &
                    (df_links['TIMESTAMP'] <= end_global)].copy()
print(f"Global filter: {len(df_links)} rows remain between {start_global} and {end_global}.")

############################
# 2. DEFINE HALF-YEAR INTERVALS
############################
# We'll define half-year boundaries from 2014-01-01 to 2017-07-01
# (the final boundary is just beyond 2017-06-30).
half_years = [
    "2014-01-01",
    "2014-07-01",
    "2015-01-01",
    "2015-07-01",
    "2016-01-01",
    "2016-07-01",
    "2017-01-01",
    "2017-07-01"  # up to but not including 07-01 means effectively 06-30 as the last day
]

############################
# 3. FUNCTIONS
############################
def split_liwc_columns(df):
    """Extract numeric LIWC_Body and LIWC_Health from the comma-separated PROPERTIES column."""
    df = df.copy()
    props_split = df['PROPERTIES'].str.split(',', expand=True)
    props_split = props_split.apply(pd.to_numeric, errors='coerce')

    # columns #68 => LIWC_Body, #69 => LIWC_Health (adjust if needed)
    df['LIWC_Body'] = props_split[68]
    df['LIWC_Health'] = props_split[69]
    return df

def compute_node_liwc_means(df):
    """
    Return a dict: node -> {LIWC_Body: x, LIWC_Health: y}
    by averaging source/target edge values for each node.
    """
    body_cols = ['LIWC_Body', 'LIWC_Health']
    src_stats = df.groupby('SOURCE_SUBREDDIT')[body_cols].mean().rename_axis('node')
    tgt_stats = df.groupby('TARGET_SUBREDDIT')[body_cols].mean().rename_axis('node')
    combined = pd.concat([src_stats, tgt_stats])
    combined = combined.groupby('node')[body_cols].mean()
    return combined.to_dict('index')

def had_healthy_neighbor(node, healthy_set, graph_undirected):
    """Check if node has at least one neighbor in healthy_set."""
    if node not in graph_undirected:
        return False
    return any((nbr in healthy_set) for nbr in graph_undirected.neighbors(node))

############################
# 4. LOOP OVER EACH CONSECUTIVE PAIR OF HALF-YEAR WINDOWS
############################

threshold = 0.01  # "healthy-lifestyle" threshold

# We'll store results in lists for plotting
window_labels = []
prob_with_list = []
prob_without_list = []
p_values_list = []

# For i in [0..len(half_years)-2), define T1 = (i,i+1), T2 = (i+1,i+2)
# We'll stop at len(half_years)-2 because T2 needs i+2 to exist
for i in range(len(half_years) - 2):
    start_t1 = half_years[i]
    end_t1   = half_years[i+1]
    start_t2 = half_years[i+1]
    end_t2   = half_years[i+2]

    # Label for this window, e.g. "2014-01-01 -> 2014-07-01 / 2014-07-01 -> 2015-01-01"
    label = f"{start_t1}–{end_t1} / {start_t2}–{end_t2}"
    window_labels.append(label)

    # 4a. Filter data for T1 and T2
    df_t1 = df_links[(df_links['TIMESTAMP'] >= start_t1) &
                     (df_links['TIMESTAMP'] < end_t1)].copy()

    df_t2 = df_links[(df_links['TIMESTAMP'] >= start_t2) &
                     (df_links['TIMESTAMP'] < end_t2)].copy()

    print(f"\n--- WINDOW #{i+1} ---")
    print(f"T1: {start_t1} to {end_t1} -> {len(df_t1)} rows")
    print(f"T2: {start_t2} to {end_t2} -> {len(df_t2)} rows")

    # 4b. Extract LIWC columns & filter
    df_t1 = split_liwc_columns(df_t1)
    df_t2 = split_liwc_columns(df_t2)

    df_t1 = df_t1[(df_t1['LIWC_Body'] > 0) | (df_t1['LIWC_Health'] > 0)]
    df_t2 = df_t2[(df_t2['LIWC_Body'] > 0) | (df_t2['LIWC_Health'] > 0)]

    print(f"After LIWC filter => T1: {len(df_t1)}, T2: {len(df_t2)}")

    # 4c. Build directed graphs
    G_t1_directed = nx.from_pandas_edgelist(
        df_t1,
        source='SOURCE_SUBREDDIT',
        target='TARGET_SUBREDDIT',
        create_using=nx.DiGraph()
    )

    G_t2_directed = nx.from_pandas_edgelist(
        df_t2,
        source='SOURCE_SUBREDDIT',
        target='TARGET_SUBREDDIT',
        create_using=nx.DiGraph()
    )

    print(f"T1 Graph => {G_t1_directed.number_of_nodes()} nodes, {G_t1_directed.number_of_edges()} edges")
    print(f"T2 Graph => {G_t2_directed.number_of_nodes()} nodes, {G_t2_directed.number_of_edges()} edges")

    # 4d. Compute node-level LIWC means
    node_liwc_t1 = compute_node_liwc_means(df_t1)
    node_liwc_t2 = compute_node_liwc_means(df_t2)

    # 4e. Mark "healthy-lifestyle" nodes at T1 and T2
    healthy_t1 = set()
    for node, vals in node_liwc_t1.items():
        avg_val = 0.5 * (vals['LIWC_Body'] + vals['LIWC_Health'])
        if avg_val >= threshold:
            healthy_t1.add(node)

    healthy_t2 = set()
    for node, vals in node_liwc_t2.items():
        avg_val = 0.5 * (vals['LIWC_Body'] + vals['LIWC_Health'])
        if avg_val >= threshold:
            healthy_t2.add(node)

    adopters = healthy_t2.difference(healthy_t1)
    print(f"Healthy T1: {len(healthy_t1)}, Healthy T2: {len(healthy_t2)}, New adopters: {len(adopters)}")

    # 4f. Check neighbor effect
    G_t1_undirected = G_t1_directed.to_undirected()

    not_healthy_t1 = set(G_t1_undirected.nodes()) - healthy_t1
    with_neighbor = set()
    without_neighbor = set()

    for node in not_healthy_t1:
        if had_healthy_neighbor(node, healthy_t1, G_t1_undirected):
            with_neighbor.add(node)
        else:
            without_neighbor.add(node)

    adopters_with_neighbor = len(adopters.intersection(with_neighbor))
    adopters_without_neighbor = len(adopters.intersection(without_neighbor))

    prob_with = adopters_with_neighbor / len(with_neighbor) if with_neighbor else 0
    prob_without = adopters_without_neighbor / len(without_neighbor) if without_neighbor else 0

    print(f"Nodes with healthy neighbor: {len(with_neighbor)}")
    print(f"Nodes without healthy neighbor: {len(without_neighbor)}")
    print(f"Adopters (with neighbor): {adopters_with_neighbor}")
    print(f"Adopters (no neighbor): {adopters_without_neighbor}")
    print(f"Probability with neighbor: {prob_with:.4f}")
    print(f"Probability no neighbor:   {prob_without:.4f}")

    # 4g. Chi-square test
    obs = np.array([
        [adopters_with_neighbor,        len(with_neighbor) - adopters_with_neighbor],
        [adopters_without_neighbor,     len(without_neighbor) - adopters_without_neighbor]
    ])
    chi2, p_val, dof, exp = chi2_contingency(obs, correction=False)
    print(f"Chi2 = {chi2:.4f}, p-value = {p_val:.4e}, dof = {dof}")

    # Store for plotting
    prob_with_list.append(prob_with)
    prob_without_list.append(prob_without_list)
    p_values_list.append(p_val)

    # Correction in the code: we must store "prob_without" in prob_without_list, not prob_without_list in prob_without_list.
    # We'll fix that below. (See note at the end.)

############################
# 5. CREATE A COMPARATIVE PLOT
############################

# We'll make a line plot for "prob_with" and "prob_without" across each half-year pair
# but we must fix the code snippet above where we inadvertently appended "prob_without_list" to itself.

# Correction for storing results:
# We'll do that properly here:
# (We've used prob_with_list, prob_without_list, p_values_list above,
# but there is a small bug in the code. We'll fix it now.)
# We rewrite the final part with the correct approach:

# Let's define the lists again, but properly:
# (We put the entire loop again with the corrected line.)

print("\nRe-running the loop with corrected data storage for the final plot...\n")
window_labels = []
prob_with_list = []
prob_without_list = []
p_values_list = []

half_years = [
    "2014-01-01",
    "2014-07-01",
    "2015-01-01",
    "2015-07-01",
    "2016-01-01",
    "2016-07-01",
    "2017-01-01",
    "2017-07-01"
]

for i in range(len(half_years) - 2):
    start_t1 = half_years[i]
    end_t1   = half_years[i+1]
    start_t2 = half_years[i+1]
    end_t2   = half_years[i+2]
    label = f"{start_t1}–{end_t1} / {start_t2}–{end_t2}"
    window_labels.append(label)

    # Filter data
    df_t1 = df_links[(df_links['TIMESTAMP'] >= start_t1) &
                     (df_links['TIMESTAMP'] < end_t1)].copy()

    df_t2 = df_links[(df_links['TIMESTAMP'] >= start_t2) &
                     (df_links['TIMESTAMP'] < end_t2)].copy()

    # LIWC
    df_t1 = split_liwc_columns(df_t1)
    df_t2 = split_liwc_columns(df_t2)

    df_t1 = df_t1[(df_t1['LIWC_Body'] > 0) | (df_t1['LIWC_Health'] > 0)]
    df_t2 = df_t2[(df_t2['LIWC_Body'] > 0) | (df_t2['LIWC_Health'] > 0)]

    # Graph
    G_t1_directed = nx.from_pandas_edgelist(df_t1,
        source='SOURCE_SUBREDDIT', target='TARGET_SUBREDDIT', create_using=nx.DiGraph())
    G_t1_undirected = G_t1_directed.to_undirected()

    # Node means
    node_liwc_t1 = compute_node_liwc_means(df_t1)
    healthy_t1 = set()
    for node, vals in node_liwc_t1.items():
        avg_val = 0.5 * (vals['LIWC_Body'] + vals['LIWC_Health'])
        if avg_val >= threshold:
            healthy_t1.add(node)

    node_liwc_t2 = compute_node_liwc_means(df_t2)
    healthy_t2 = set()
    for node, vals in node_liwc_t2.items():
        avg_val = 0.5 * (vals['LIWC_Body'] + vals['LIWC_Health'])
        if avg_val >= threshold:
            healthy_t2.add(node)

    # Adopters
    adopters = healthy_t2.difference(healthy_t1)
    not_healthy_t1 = set(G_t1_undirected.nodes()) - healthy_t1

    with_neighbor = set()
    without_neighbor = set()

    for node in not_healthy_t1:
        if had_healthy_neighbor(node, healthy_t1, G_t1_undirected):
            with_neighbor.add(node)
        else:
            without_neighbor.add(node)

    adopters_with_neighbor = len(adopters.intersection(with_neighbor))
    adopters_without_neighbor = len(adopters.intersection(without_neighbor))

    prob_with = adopters_with_neighbor / len(with_neighbor) if with_neighbor else 0
    prob_without = adopters_without_neighbor / len(without_neighbor) if without_neighbor else 0

    # Chi-square
    obs = np.array([
        [adopters_with_neighbor,        len(with_neighbor) - adopters_with_neighbor],
        [adopters_without_neighbor,     len(without_neighbor) - adopters_without_neighbor]
    ])
    chi2, p_val, dof, exp = chi2_contingency(obs, correction=False)

    # Store
    prob_with_list.append(prob_with)
    prob_without_list.append(prob_without)
    p_values_list.append(p_val)

# Now we do a plot
x = np.arange(len(window_labels))

plt.figure(figsize=(10, 6))
plt.plot(x, prob_with_list, marker='o', label='Probability of Adoption (Had Healthy Neighbor)')
plt.plot(x, prob_without_list, marker='o', label='Probability of Adoption (No Healthy Neighbor)')

plt.xticks(x, window_labels, rotation=30, ha='right')
plt.title("Neighbor Effect Adoption Probability\nHalf-Year Windows from 2014-01-01 to 2017-06-30")
plt.ylabel("Probability of Adopting 'Healthy-Lifestyle'")
plt.legend()
plt.tight_layout()
plt.show()

# Optionally, you could also plot p-values or print them
print("\nP-Values for each window pair:")
for label, pv in zip(window_labels, p_values_list):
    print(f"{label}: p-value = {pv:.4e}")

print("\nDone.")
