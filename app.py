"""
app.py

Customer DNA -- business-facing customer segmentation (K-Means, with
business labels) plus a technical Analytics tab covering DBSCAN,
Hierarchical, GMM, and a comparative analysis.

Single flat file, plain functions -- no shared abstractions between
sections. Some repetition is intentional in favor of reading this file
top-to-bottom without jumping between modules.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster

st.set_page_config(page_title="Customer DNA", page_icon="🧬", layout="wide")

# =====================================================================
# Theme -- light, restrained, one confident accent color (deep navy)
# kept deliberately separate from the segment palette below, so color
# only ever means "which segment" and never doubles as generic UI
# chrome. Base palette (light/navy/Inter) lives in .streamlit/config.toml;
# this is the small set of safe, stable data-testid-based CSS tweaks
# config.toml can't reach on its own.
# =====================================================================

BRAND_COLOR = "#1B2A4A"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}

    [data-testid="stMetricValue"] {{
        font-weight: 700;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.78rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: #6B7280;
    }}

    section[data-testid="stSidebar"] {{
        border-right: 1px solid #E7E9EE;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 12px !important;
    }}

    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        color: {BRAND_COLOR} !important;
        font-weight: 600;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {BRAND_COLOR} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_brand_header():
    """Styled wordmark used in place of st.title()+st.caption() -- the
    'DNA' half picks up the brand accent, everything else stays neutral."""
    st.markdown(
        f"""
        <div style="display:flex; align-items:baseline; gap:0.6rem; margin-bottom:0;">
            <span style="font-size:2.1rem;">🧬</span>
            <h1 style="margin:0; font-weight:700; letter-spacing:-0.02em;">
                Customer <span style="color:{BRAND_COLOR};">DNA</span>
            </h1>
        </div>
        <p style="color:#6B7280; font-size:1.05rem; margin-top:0.15rem; margin-bottom:1rem;">
            Decode who your customers really are — RFM analysis + machine-learning segmentation
        </p>
        """,
        unsafe_allow_html=True,
    )


RFM_COLUMNS = ["Recency", "Frequency", "MonetaryValue"]

# Hardcoded schema -- Online Retail II column conventions.
CUSTOMER_ID_COL = "Customer ID"
INVOICE_COL = "Invoice"
INVOICE_DATE_COL = "InvoiceDate"
STOCK_CODE_COL = "StockCode"
QUANTITY_COL = "Quantity"
PRICE_COL = "Price"

# =====================================================================
# Design system -- color theory applied to the business meaning of
# each metric / segment, used consistently everywhere a chart, badge,
# or card refers to it.
# =====================================================================

# Recency = time/waiting -> calm, cool blue. Frequency = activity/energy
# -> warm amber. MonetaryValue = money -> classic money-green. Distinct
# hues so the three read apart even for a colorblind viewer.
RFM_COLORS = {
    "Recency": "#4A6FA5",
    "Frequency": "#E8871E",
    "MonetaryValue": "#2E8B57",
}

# Core segments (from assign_business_labels, k=4) + outlier segments
# (from get_outlier_segments). Each hue is chosen for what the segment
# means to the business, not just to look different from its neighbors:
#   REWARD     gold    -- premium / top value
#   RETAIN     blue    -- trust / stability
#   NURTURE    green   -- growth potential
#   RE-ENGAGE  red     -- urgency / at risk
#   PAMPER     purple  -- luxury / indulgence
#   UPSELL     teal    -- opportunity
#   DELIGHT    rose    -- VIP / celebration
SEGMENT_COLORS = {
    "REWARD": "#D4A017",
    "RETAIN": "#2C5F8A",
    "NURTURE": "#3F9142",
    "RE-ENGAGE": "#D64550",
    "PAMPER": "#8E44AD",
    "UPSELL": "#17A2A2",
    "DELIGHT": "#C9184A",
}


# =====================================================================
# Data pipeline
# =====================================================================

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file, sheet_name=0)


def clean_transaction_data(df):
    df = df.copy()
    required = [INVOICE_COL, STOCK_CODE_COL, CUSTOMER_ID_COL, PRICE_COL, QUANTITY_COL, INVOICE_DATE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df[INVOICE_COL] = df[INVOICE_COL].astype(str)
    df = df[df[INVOICE_COL].str.match(r"^\d{6}$", na=False)]

    df[STOCK_CODE_COL] = df[STOCK_CODE_COL].astype(str)
    stock_mask = (
        df[STOCK_CODE_COL].str.match(r"^\d{5}$", na=False)
        | df[STOCK_CODE_COL].str.match(r"^\d{5}[a-zA-Z]+$", na=False)
    )
    df = df[stock_mask]

    df = df.dropna(subset=[CUSTOMER_ID_COL])
    df = df[df[PRICE_COL] > 0].copy()

    df[INVOICE_DATE_COL] = pd.to_datetime(df[INVOICE_DATE_COL], errors="coerce")
    df = df.dropna(subset=[INVOICE_DATE_COL])

    df["SalesLineTotal"] = df[QUANTITY_COL] * df[PRICE_COL]
    return df


def build_customer_features(df):
    customer_df = (
        df.groupby(CUSTOMER_ID_COL, as_index=False)
        .agg(
            MonetaryValue=("SalesLineTotal", "sum"),
            Frequency=(INVOICE_COL, "nunique"),
            LastInvoiceDate=(INVOICE_DATE_COL, "max"),
        )
    )
    max_invoice_date = customer_df["LastInvoiceDate"].max()
    customer_df["Recency"] = (max_invoice_date - customer_df["LastInvoiceDate"]).dt.days
    return customer_df


def find_outliers(customer_df):
    """IQR on Frequency + MonetaryValue only -- NOT Recency (see notes)."""
    flags = pd.DataFrame(index=customer_df.index)
    for feature in ["MonetaryValue", "Frequency"]:
        q1 = customer_df[feature].quantile(0.25)
        q3 = customer_df[feature].quantile(0.75)
        iqr = q3 - q1
        flags[f"{feature}_outlier"] = (
            (customer_df[feature] > q3 + 1.5 * iqr) | (customer_df[feature] < q1 - 1.5 * iqr)
        )
    monetary_outliers = customer_df[flags["MonetaryValue_outlier"]].copy()
    frequency_outliers = customer_df[flags["Frequency_outlier"]].copy()
    non_outliers = customer_df[~flags.any(axis=1)].copy()
    return monetary_outliers, frequency_outliers, non_outliers


def scale_features(df, columns=RFM_COLUMNS):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[columns])
    return pd.DataFrame(scaled, index=df.index, columns=columns), scaler


# =====================================================================
# Elbow / parameter-selection helpers
# =====================================================================

def find_elbow_by_distance(x_values, y_values):
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    dx = x_norm - p1[0]
    dy = y_norm - p1[1]
    # 2D cross product magnitude (line_vec x point_vec), computed directly --
    # np.cross() no longer accepts 2D vectors as of NumPy 2.0.
    cross = line_vec[0] * dy - line_vec[1] * dx
    distances = np.abs(cross) / (line_len + 1e-12)

    elbow_idx = int(np.argmax(distances))
    return int(x[elbow_idx])


def find_merge_gap_k(clusters_before_merge, merge_distance):
    df = pd.DataFrame({"clusters_before_merge": clusters_before_merge, "merge_distance": merge_distance}) \
        .sort_values("clusters_before_merge", ascending=False).reset_index(drop=True)
    gaps = df["merge_distance"].diff().fillna(0)
    jump_idx = int(gaps.idxmax())
    if jump_idx == 0:
        return int(df.iloc[0]["clusters_before_merge"])
    return int(df.iloc[jump_idx - 1]["clusters_before_merge"])


def find_eps_plateau(sweep_df, min_clusters=5, max_clusters=9):
    in_band = sweep_df["n_clusters"].between(min_clusters, max_clusters)
    best_start, best_len, run_start = None, 0, None
    for i, ok in enumerate(in_band.tolist() + [False]):
        if ok and run_start is None:
            run_start = i
        elif not ok and run_start is not None:
            run_len = i - run_start
            if run_len > best_len:
                best_len, best_start = run_len, run_start
            run_start = None
    if best_start is None:
        return float(sweep_df.loc[sweep_df["n_clusters"].idxmax(), "eps"])
    band = sweep_df.iloc[best_start: best_start + best_len]
    return float(band.iloc[len(band) // 2]["eps"])


# =====================================================================
# Clustering methods
# =====================================================================

def run_kmeans(df, k):
    scaled_df, _ = scale_features(df)
    model = KMeans(n_clusters=k, random_state=42, max_iter=1000, n_init=10)
    labels = model.fit_predict(scaled_df)
    result = df.copy()
    result["Cluster"] = labels
    return result, scaled_df


def suggest_kmeans_k(df, k_range=range(2, 13)):
    scaled_df, _ = scale_features(df)
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, max_iter=1000, n_init=10)
        labels = model.fit_predict(scaled_df)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette_score(scaled_df, labels)})
    return pd.DataFrame(rows)


def run_dbscan(df, eps, min_samples):
    scaled_df, _ = scale_features(df)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(scaled_df)
    result = df.copy()
    result["Cluster"] = labels
    return result


def suggest_dbscan_eps_sweep(df, min_samples=5, eps_min=0.05, eps_max=0.30, eps_step=0.01):
    scaled_df, _ = scale_features(df)
    eps_values = np.arange(eps_min, eps_max + 1e-9, eps_step)
    rows = []
    for eps in eps_values:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(scaled_df)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(list(labels).count(-1))
        rows.append({"eps": round(float(eps), 2), "n_clusters": n_clusters,
                      "n_noise": n_noise, "noise_pct": round(n_noise / len(labels) * 100, 2)})
    return pd.DataFrame(rows)


def run_hierarchical(df, k):
    scaled_df, _ = scale_features(df)
    Z = linkage(scaled_df, method="ward")
    labels = fcluster(Z, k, criterion="maxclust")
    result = df.copy()
    result["Cluster"] = labels
    return result


def suggest_hierarchical_gap(df, last_n=15):
    scaled_df, _ = scale_features(df)
    Z = linkage(scaled_df, method="ward")
    n_samples = len(scaled_df)
    n_merges = Z.shape[0]
    rows = []
    for i in range(max(0, n_merges - last_n), n_merges):
        rows.append({"clusters_before_merge": n_samples - i, "merge_distance": Z[i, 2]})
    return pd.DataFrame(rows)


def run_gmm(df, k):
    scaled_df, _ = scale_features(df)
    model = GaussianMixture(n_components=k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_df)
    result = df.copy()
    result["Cluster"] = labels
    return result


def suggest_gmm_k(df, k_range=range(2, 21)):
    scaled_df, _ = scale_features(df)
    rows = []
    for k in k_range:
        model = GaussianMixture(n_components=k, random_state=42, n_init=10)
        model.fit(scaled_df)
        rows.append({"k": k, "bic": model.bic(scaled_df)})
    return pd.DataFrame(rows)


def compare_methods(results: dict):
    rows = []
    for name, labeled_df in results.items():
        non_noise = labeled_df[labeled_df["Cluster"] != -1]
        n_clusters = non_noise["Cluster"].nunique()
        n_excluded = len(labeled_df) - len(non_noise)
        if n_clusters < 2:
            score = np.nan
        else:
            scaled_df, _ = scale_features(non_noise)
            score = silhouette_score(scaled_df, non_noise["Cluster"])
        rows.append({"method": name, "silhouette_score": score, "n_clusters": n_clusters, "n_excluded": n_excluded})
    return pd.DataFrame(rows)


# =====================================================================
# Business labeling (K-Means only -- the primary/customer-facing method)
# =====================================================================

def assign_business_labels(clustered_df, k):
    """Labels from cluster profiles, not assumed cluster-ID order."""
    labels = {c: f"CLUSTER {c}" for c in sorted(clustered_df["Cluster"].unique())}
    if k != 4:
        return labels
    profile = clustered_df.groupby("Cluster")[RFM_COLUMNS].mean()
    z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, 1)
    z["ValueScore"] = z["MonetaryValue"] + z["Frequency"] - z["Recency"]
    reward = z["ValueScore"].idxmax()
    reengage = z["ValueScore"].idxmin()
    remaining = [c for c in z.index if c not in {reward, reengage}]
    if remaining:
        nurture = min(remaining, key=lambda c: z.loc[c, "MonetaryValue"] + z.loc[c, "Frequency"])
        retain = next(c for c in remaining if c != nurture)
    else:
        nurture = retain = reward
    labels[reward] = "REWARD"
    labels[retain] = "RETAIN"
    labels[nurture] = "NURTURE"
    labels[reengage] = "RE-ENGAGE"
    return labels


def get_outlier_segments(monetary_outliers, frequency_outliers):
    overlap = monetary_outliers.index.intersection(frequency_outliers.index)
    monetary_only = monetary_outliers.drop(overlap).copy()
    frequency_only = frequency_outliers.drop(overlap).copy()
    both = monetary_outliers.loc[overlap].copy()
    monetary_only["Cluster"] = -1
    frequency_only["Cluster"] = -2
    both["Cluster"] = -3
    return pd.concat([monetary_only, frequency_only, both])


OUTLIER_LABELS = {-1: "PAMPER", -2: "UPSELL", -3: "DELIGHT"}

BUSINESS_INTERPRETATIONS = {
    "RETAIN": "High-value customers with consistent purchasing behavior. Focus on loyalty, personalized communication and sustained engagement.",
    "RE-ENGAGE": "Lower-value or infrequent customers with weaker recent engagement. Use targeted campaigns, reminders and time-bound offers.",
    "NURTURE": "Recently active customers with relatively lower value and frequency. Use recommendations, incentives and cross-sell opportunities to grow the relationship.",
    "REWARD": "Highly valuable customers with strong spending and purchasing frequency. Prioritize recognition, exclusivity and premium loyalty benefits.",
    "PAMPER": "High-spending outlier customers. Treat them as strategically important and investigate what drives their unusually high value.",
    "UPSELL": "High-frequency outlier customers. Explore bundles, cross-sell and upsell opportunities.",
    "DELIGHT": "Extreme-value customers combining unusually high spending and frequency. Consider VIP-style experiences and retention strategies.",
}

# Richer strategy content for the Strategies tab: an icon, a punchy
# headline, and concrete, active-voice next actions per segment.
STRATEGY_DETAILS = {
    "REWARD": {
        "icon": "🏆",
        "headline": "Your top-tier customers -- high spend, high frequency, recently active.",
        "actions": [
            "Give early access to new arrivals and limited releases",
            "Assign a dedicated contact for priority support",
            "Create an invite-only loyalty tier with premium perks",
        ],
    },
    "RETAIN": {
        "icon": "🤝",
        "headline": "Reliable, steady spenders -- the backbone of consistent revenue.",
        "actions": [
            "Send personalized thank-yous and loyalty rewards",
            "Recommend products based on their purchase history",
            "Keep them engaged with regular, low-pressure check-ins",
        ],
    },
    "NURTURE": {
        "icon": "🌱",
        "headline": "Recently active, but still building spend and frequency.",
        "actions": [
            "Offer a first-purchase-anniversary discount",
            "Cross-sell complementary products",
            "Build the habit with an onboarding-style email sequence",
        ],
    },
    "RE-ENGAGE": {
        "icon": "⏰",
        "headline": "Drifting away -- low recent activity, at risk of churning.",
        "actions": [
            "Launch a win-back campaign with a time-limited offer",
            "Ask for feedback on why they stopped purchasing",
            "Retarget with reminders of previously viewed items",
        ],
    },
    "PAMPER": {
        "icon": "💎",
        "headline": "Unusually high spenders -- find out what makes them special.",
        "actions": [
            "Interview a sample to learn what drives their spend",
            "Offer concierge-style service or account management",
            "Reach out proactively to protect the relationship",
        ],
    },
    "UPSELL": {
        "icon": "📈",
        "headline": "Shop often but spend less per visit -- ready for bigger baskets.",
        "actions": [
            "Bundle frequently-bought items at a discount",
            "Suggest premium alternatives at checkout",
            "Introduce a subscribe-and-save option",
        ],
    },
    "DELIGHT": {
        "icon": "✨",
        "headline": "Extreme value on both spend and frequency -- your best customers, full stop.",
        "actions": [
            "Roll out white-glove, VIP-style experiences",
            "Feature them in loyalty or referral programs",
            "Check in proactively to prevent any churn risk",
        ],
    },
}


# =====================================================================
# Charts
# =====================================================================

def plot_3d_scatter(labeled_df, title, color_col="Cluster", use_log_transform=False,
                     axis_columns=None, color_discrete_map=None):
    plot_df = labeled_df.copy()
    x, y, z = axis_columns or ("Recency", "Frequency", "MonetaryValue")
    if use_log_transform:
        for col in [x, y, z]:
            plot_df[f"{col}_plot"] = np.log1p(plot_df[col])
        x_col, y_col, z_col = f"{x}_plot", f"{y}_plot", f"{z}_plot"
    else:
        x_col, y_col, z_col = x, y, z
    fig = px.scatter_3d(plot_df, x=x_col, y=y_col, z=z_col,
                         color=plot_df[color_col].astype(str),
                         color_discrete_map=color_discrete_map,
                         hover_data={x: True, y: True, z: True}, title=title)
    fig.update_layout(height=600, legend_title_text=color_col)
    return fig


def plot_cluster_distribution(labeled_df, title, group_col="Cluster", color_discrete_map=None):
    plot_df = labeled_df[labeled_df["Cluster"] != -1] if -1 in labeled_df["Cluster"].values else labeled_df
    sizes = plot_df[group_col].value_counts()
    means = plot_df.groupby(group_col)[RFM_COLUMNS].mean()
    fig = go.Figure()
    if color_discrete_map:
        bar_colors = [color_discrete_map.get(str(g), "#4A6FA5") for g in sizes.index]
        fig.add_bar(x=sizes.index.astype(str), y=sizes.values, name="Customers", marker_color=bar_colors)
    else:
        fig.add_bar(x=sizes.index.astype(str), y=sizes.values, name="Customers")
    for col in RFM_COLUMNS:
        fig.add_scatter(x=means.index.astype(str), y=means[col], name=f"Mean {col}", yaxis="y2", mode="lines+markers")
    fig.update_layout(title=title, height=400, yaxis=dict(title="Customers"),
                       yaxis2=dict(title="Mean feature value", overlaying="y", side="right"))
    return fig


def plot_feature_distribution_by_cluster(labeled_df, feature, plot_type="violin"):
    fig, ax = plt.subplots(figsize=(8, 4))
    clusters = sorted(labeled_df["Cluster"].unique())
    data = [labeled_df[labeled_df["Cluster"] == c][feature].values for c in clusters]
    # One distinct color per cluster (not one flat color for all of them).
    palette = plt.get_cmap("tab10").colors if len(clusters) <= 10 else plt.get_cmap("tab20").colors
    colors = [palette[i % len(palette)] for i in range(len(clusters))]
    if plot_type == "violin":
        parts = ax.violinplot(data, showmeans=True)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.75)
        for key in ("cmeans", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color("#444444")
    else:
        for i, (d, color) in enumerate(zip(data, colors), start=1):
            jitter = np.random.normal(loc=i, scale=0.04, size=len(d))
            ax.scatter(jitter, d, alpha=0.6, s=10, color=color)
    ax.set_xticks(range(1, len(clusters) + 1))
    ax.set_xticklabels([str(c) for c in clusters])
    ax.set_xlabel("Cluster")
    ax.set_ylabel(feature)
    return fig


def plot_single_box(series, feature, color):
    """One vertical box for one feature -- matches the notebook's own
    boxplot layout exactly (single box per feature, linear scale, value
    on the y-axis). A thin box with a long tail of outlier dots on
    Frequency/MonetaryValue is expected on the 'before' view -- that's
    just what the real distribution looks like, not a rendering bug."""
    fig = go.Figure()
    fig.add_trace(go.Box(y=series, name=feature, marker_color=color, boxmean=True))
    fig.update_layout(
        title=f"{feature} Boxplot", height=380, showlegend=False,
        xaxis=dict(showticklabels=False), yaxis_title=feature,
    )
    return fig


# =====================================================================
# Sidebar
# =====================================================================

st.sidebar.title("🧬 Customer DNA")
st.sidebar.caption("RFM Analysis + K-Means Clustering")
st.sidebar.markdown("### Settings")
remove_outliers = st.sidebar.checkbox("Remove Outliers (IQR method)", value=True)
auto_k = st.sidebar.checkbox("Auto-detect Best K", value=True, help="Uses the elbow method on the inertia curve.")
manual_k = st.sidebar.slider("Number of Clusters (K) — manual override", 2, 10, 4, disabled=auto_k)

st.sidebar.markdown("### Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Your Dataset (.xlsx or .csv)", type=["xlsx", "xls", "csv"])

if uploaded_file is None:
    render_brand_header()
    st.info("Upload your dataset in the sidebar to get started.")
    st.stop()

# =====================================================================
# Pipeline
# =====================================================================

try:
    with st.spinner("Cleaning data and building customer features..."):
        raw_df = load_data(uploaded_file)
        cleaned_df = clean_transaction_data(raw_df)
        customer_df = build_customer_features(cleaned_df)
        monetary_outliers, frequency_outliers, non_outliers_df = find_outliers(customer_df)
        aggregated_df = customer_df  # full population, used by DBSCAN and when outlier removal is off
        kmeans_input_df = non_outliers_df if remove_outliers else aggregated_df
except Exception as exc:
    st.error(f"The file could not be processed: {exc}")
    st.stop()

# K-Means auto-runs -- it's the primary business path, cheap enough at this scale to not gate behind a button.
with st.spinner("Segmenting customers..."):
    kmeans_rec_df = suggest_kmeans_k(kmeans_input_df)
    recommended_k = find_elbow_by_distance(kmeans_rec_df["k"], kmeans_rec_df["inertia"])
    active_k = recommended_k if auto_k else manual_k
    kmeans_result, kmeans_scaled_df = run_kmeans(kmeans_input_df, active_k)
    kmeans_labels = assign_business_labels(kmeans_result, active_k)
    kmeans_result["ClusterLabel"] = kmeans_result["Cluster"].map(kmeans_labels)
    # Scaled RFM columns, aligned to kmeans_result's index, for the "true" view
    # of what K-Means actually clustered on (see Segments tab).
    kmeans_result["Recency_scaled"] = kmeans_scaled_df["Recency"]
    kmeans_result["Frequency_scaled"] = kmeans_scaled_df["Frequency"]
    kmeans_result["MonetaryValue_scaled"] = kmeans_scaled_df["MonetaryValue"]

    outlier_df = get_outlier_segments(monetary_outliers, frequency_outliers)
    if len(outlier_df) > 0:
        outlier_df["ClusterLabel"] = outlier_df["Cluster"].map(OUTLIER_LABELS)
    full_df = pd.concat([kmeans_result, outlier_df], axis=0) if remove_outliers else kmeans_result

render_brand_header()

tab_names = ["Overview", "Data Quality", "Customer Metrics", "Segments",
             "Strategies", "Find a Customer", "Analytics"]
tabs = st.tabs(tab_names)

# --- Overview ---
with tabs[0]:
    st.subheader("Raw Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{len(raw_df):,}")
    c2.metric("Total Columns", f"{raw_df.shape[1]}")
    c3.metric("Unique Customers", f"{customer_df.shape[0]:,}")
    c4.metric("Unique Invoices", f"{cleaned_df[INVOICE_COL].nunique():,}")
    st.subheader("Sample Records")
    st.dataframe(raw_df.head(20), use_container_width=True, hide_index=True)

# --- Data Quality ---
with tabs[1]:
    st.subheader("Data Quality & Cleaning")

    # Unique outlier-customer count (dedup: a customer flagged on both
    # Monetary and Frequency counts once, not twice).
    outlier_customer_count = len(monetary_outliers.index.union(frequency_outliers.index))

    q1, q2, q3 = st.columns(3)
    q1.metric("Rows removed", f"{len(raw_df) - len(cleaned_df):,}")
    q2.metric("Retention rate", f"{len(cleaned_df) / len(raw_df) * 100:.2f}%")
    q3.metric("Outlier customers", f"{outlier_customer_count:,}")

    col_funnel, col_donut = st.columns([2, 1])
    with col_funnel:
        funnel_fig = go.Figure(go.Funnel(
            y=["Raw transaction rows", "Cleaned transaction rows", "Unique customers", "Core customers"],
            x=[len(raw_df), len(cleaned_df), customer_df.shape[0], len(non_outliers_df)],
            textinfo="value+percent initial",
            marker={"color": ["#8FB3D9", "#5B8DBE", "#2C5F8A", "#2E8B57"]},
        ))
        funnel_fig.update_layout(title="From raw rows to core customers", height=380)
        st.plotly_chart(funnel_fig, use_container_width=True)
    with col_donut:
        donut_df = pd.DataFrame({
            "Group": ["Core customers", "Outlier customers"],
            "Count": [len(non_outliers_df), outlier_customer_count],
        })
        donut_fig = px.pie(
            donut_df, names="Group", values="Count", hole=0.55,
            color="Group", color_discrete_map={"Core customers": "#2E8B57", "Outlier customers": "#E8871E"},
            title="Customer base split",
        )
        donut_fig.update_layout(height=380)
        st.plotly_chart(donut_fig, use_container_width=True)

    st.markdown("##### Cleaning rules applied")
    r1, r2, r3, r4 = st.columns(4)
    r1.markdown("**🧾 Invoice**  \nExactly 6 digits")
    r2.markdown("**🏷️ Stock Code**  \n5 digits, optional letters")
    r3.markdown("**👤 Customer ID**  \nMissing rows dropped")
    r4.markdown("**💰 Price**  \nMust be greater than 0")
    
# --- Customer Metrics ---
with tabs[2]:
    st.subheader("Customer-Level Metrics (RFM)")
    st.caption("Recency = days since last invoice · Frequency = distinct invoices · Monetary Value = total spend")

    cols = st.columns(3)
    for col, feature in zip(cols, RFM_COLUMNS):
        with col:
            fig = px.histogram(customer_df, x=feature, nbins=30, title=feature,
                                color_discrete_sequence=[RFM_COLORS[feature]])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Before vs. after outlier removal")
    st.caption(
        "Same three features, comparing the full customer base to the core (non-outlier) set the "
        "segmentation is built on."
    )

    st.markdown("**Before removal**")
    b1, b2, b3 = st.columns(3)
    b1.plotly_chart(plot_single_box(customer_df["Recency"], "Recency", RFM_COLORS["Recency"]), use_container_width=True)
    b2.plotly_chart(plot_single_box(customer_df["Frequency"], "Frequency", RFM_COLORS["Frequency"]), use_container_width=True)
    b3.plotly_chart(plot_single_box(customer_df["MonetaryValue"], "MonetaryValue", RFM_COLORS["MonetaryValue"]), use_container_width=True)

    st.markdown("**After removal**")
    a1, a2, a3 = st.columns(3)
    a1.plotly_chart(plot_single_box(non_outliers_df["Recency"], "Recency", RFM_COLORS["Recency"]), use_container_width=True)
    a2.plotly_chart(plot_single_box(non_outliers_df["Frequency"], "Frequency", RFM_COLORS["Frequency"]), use_container_width=True)
    a3.plotly_chart(plot_single_box(non_outliers_df["MonetaryValue"], "MonetaryValue", RFM_COLORS["MonetaryValue"]), use_container_width=True)

# --- Segments ---
with tabs[3]:
    st.subheader("Customer Segments")
    st.caption(f"K-Means identified {active_k} customer segments"
               f"{' (auto-selected via elbow method)' if auto_k else ' (manual selection)'}.")

    st.plotly_chart(
        plot_3d_scatter(
            kmeans_result, "Customer Segments — Scaled RFM (what K-Means actually clustered on)",
            color_col="ClusterLabel",
            axis_columns=("Recency_scaled", "Frequency_scaled", "MonetaryValue_scaled"),
            color_discrete_map=SEGMENT_COLORS,
        ),
        use_container_width=True,
    )

    counts = full_df["ClusterLabel"].value_counts().rename_axis("Segment").reset_index(name="Customers")
    fig = px.bar(counts, x="Segment", y="Customers", text="Customers", title="Segment Sizes",
                 color="Segment", color_discrete_map=SEGMENT_COLORS)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    profile = full_df.groupby("ClusterLabel")[RFM_COLUMNS].mean().sort_values("MonetaryValue", ascending=False)
    st.dataframe(profile.style.format("{:.2f}"), use_container_width=True)

    with st.expander("How was K chosen?"):
        st.info(f"k={recommended_k} recommended via the elbow of the inertia curve.")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(kmeans_rec_df["k"], kmeans_rec_df["inertia"], marker="o")
        ax1.axvline(active_k, color="red", linestyle="--")
        ax1.set_xlabel("k"); ax1.set_ylabel("inertia")
        ax2.plot(kmeans_rec_df["k"], kmeans_rec_df["silhouette"], marker="o", color="orange")
        ax2.axvline(active_k, color="red", linestyle="--")
        ax2.set_xlabel("k"); ax2.set_ylabel("silhouette")
        st.pyplot(fig)
        plt.close(fig)

# --- Strategies ---
with tabs[4]:
    st.subheader("Segment → Business Actions")

    total_customers = len(full_df)
    total_revenue = full_df["MonetaryValue"].sum()

    segment_order = (
        full_df.groupby("ClusterLabel")["MonetaryValue"].sum().sort_values(ascending=False).index.tolist()
    )

    top_segment = segment_order[0]
    top_pct_customers = (full_df["ClusterLabel"] == top_segment).sum() / total_customers * 100
    top_pct_revenue = full_df.loc[full_df["ClusterLabel"] == top_segment, "MonetaryValue"].sum() / total_revenue * 100
    st.info(
        f"**{top_segment}** is just **{top_pct_customers:.1f}%** of your customers, "
        f"but drives **{top_pct_revenue:.1f}%** of total revenue — this is where the business impact concentrates."
    )

    for segment in segment_order:
        seg_df = full_df[full_df["ClusterLabel"] == segment]
        if seg_df.empty:
            continue
        n = len(seg_df)
        pct_customers = n / total_customers * 100
        pct_revenue = seg_df["MonetaryValue"].sum() / total_revenue * 100
        color = SEGMENT_COLORS.get(segment, "#4A6FA5")
        details = STRATEGY_DETAILS.get(segment)

        with st.container(border=True):
            st.markdown(
                f"<h4 style='margin-bottom:0'>"
                f"<span style='color:{color}'>{details['icon'] if details else '●'} {segment}</span>"
                f"</h4>",
                unsafe_allow_html=True,
            )
            if details:
                st.write(details["headline"])
            else:
                st.write(BUSINESS_INTERPRETATIONS.get(segment, "Review the segment profile and determine an appropriate strategy."))

            m1, m2, m3 = st.columns(3)
            m1.metric("Customers", f"{n:,}", f"{pct_customers:.1f}% of base")
            m2.metric("Revenue share", f"{pct_revenue:.1f}%")
            m3.metric("Avg. spend / customer", f"£{seg_df['MonetaryValue'].mean():,.0f}")

            if details:
                st.markdown("**Recommended actions**")
                for action in details["actions"]:
                    st.markdown(f"- {action}")

# --- Find a Customer (search) ---
with tabs[5]:
    st.subheader("Find a Customer")
    st.caption("Enter a Customer ID to see which segment they belong to.")
    customer_search = st.text_input("Customer ID", placeholder="e.g. 17850")

    if customer_search:
        search_clean = customer_search.strip()
        try:
            search_val = float(search_clean)
            match = full_df[full_df[CUSTOMER_ID_COL].astype(float) == search_val]
        except ValueError:
            match = full_df[full_df[CUSTOMER_ID_COL].astype(str) == search_clean]

        if match.empty:
            st.warning(f"No customer found with ID {customer_search}.")
        else:
            row = match.iloc[0]
            label = row["ClusterLabel"]
            color = SEGMENT_COLORS.get(label, "#4A6FA5")
            icon = STRATEGY_DETAILS.get(label, {}).get("icon", "●")

            st.markdown(
                f"<h3>Customer {customer_search} is in "
                f"<span style='color:{color}'>{icon} {label}</span></h3>",
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Recency (days)", f"{row['Recency']:.0f}")
            m2.metric("Frequency", f"{row['Frequency']:.0f}")
            m3.metric("Monetary Value", f"£{row['MonetaryValue']:,.2f}")
            st.info(BUSINESS_INTERPRETATIONS.get(label, ""))

# --- Analytics (technical: all 4 methods + comparison) ---
with tabs[6]:
    st.caption("Technical detail for methodology review -- parameter selection, alternative clustering methods, comparison.")
    sub_tabs = st.tabs(["DBSCAN", "Hierarchical", "GMM", "Comparison"])

    with sub_tabs[0]:
        st.subheader("DBSCAN")
        if st.button("Compute parameter recommendation", key="dbscan_suggest"):
            with st.spinner("Sweeping eps..."):
                sweep_df = suggest_dbscan_eps_sweep(aggregated_df, min_samples=5)
                st.session_state["dbscan_sweep"] = sweep_df
                # Pre-set the number_input's value BEFORE it's instantiated below --
                # Streamlit ignores `value=` on a widget's key once that key already
                # exists in session_state, so this is the only way a recommendation
                # can actually change what's used on the next "Run" click.
                st.session_state["dbscan_eps"] = find_eps_plateau(sweep_df, min_clusters=5, max_clusters=9)
        default_eps, default_min_samples = 0.14, 5
        if "dbscan_sweep" in st.session_state:
            sweep_df = st.session_state["dbscan_sweep"]
            chosen_eps = find_eps_plateau(sweep_df, min_clusters=5, max_clusters=9)
            default_eps = chosen_eps
            st.info(f"eps={chosen_eps:.2f} recommended -- middle of the widest stable band of 5-9 clusters.")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(sweep_df["eps"], sweep_df["n_clusters"], marker="o")
            ax.axvline(chosen_eps, color="red", linestyle="--", label=f"chosen eps={chosen_eps:.2f}")
            ax.set_xlabel("eps"); ax.set_ylabel("n_clusters"); ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        eps = st.number_input("eps", min_value=0.01, value=float(default_eps), step=0.01, key="dbscan_eps")
        min_samples = st.number_input("min_samples", min_value=1, value=default_min_samples, step=1, key="dbscan_min_samples")
        if st.button("Run DBSCAN", key="dbscan_run"):
            with st.spinner("Running DBSCAN..."):
                st.session_state["dbscan_result"] = run_dbscan(aggregated_df, eps, int(min_samples))
        if "dbscan_result" in st.session_state:
            result = st.session_state["dbscan_result"]
            n_clusters = result[result["Cluster"] != -1]["Cluster"].nunique()
            n_noise = int((result["Cluster"] == -1).sum())
            st.success(f"{n_clusters} clusters, {n_noise} noise points ({n_noise/len(result)*100:.1f}%)")
            st.plotly_chart(plot_3d_scatter(result, "DBSCAN Segments", use_log_transform=True), use_container_width=True)
            st.plotly_chart(plot_cluster_distribution(result, "DBSCAN Cluster Sizes & Feature Means"), use_container_width=True)
            feature = st.selectbox("Feature to inspect by cluster", RFM_COLUMNS, key="dbscan_feature")
            fig = plot_feature_distribution_by_cluster(result, feature, plot_type="strip")
            st.pyplot(fig)
            plt.close(fig)

    with sub_tabs[1]:
        st.subheader("Hierarchical")
        if st.button("Compute parameter recommendation", key="hier_suggest"):
            with st.spinner("Building dendrogram..."):
                st.session_state["hier_rec"] = suggest_hierarchical_gap(non_outliers_df)
                # Pre-set before the slider below is instantiated (see DBSCAN note above).
                st.session_state["hier_k"] = active_k
        default_hier_k = active_k
        if "hier_rec" in st.session_state:
            rec_df = st.session_state["hier_rec"]
            gap_k = find_merge_gap_k(rec_df["clusters_before_merge"], rec_df["merge_distance"])
            st.info(f"k={default_hier_k} recommended -- matches K-Means' k for direct comparability "
                    f"(the raw merge-distance gap alone would suggest k={gap_k}).")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(rec_df["clusters_before_merge"], rec_df["merge_distance"], marker="o")
            ax.axvline(default_hier_k, color="red", linestyle="--", label=f"chosen k={default_hier_k}")
            ax.axvline(gap_k, color="gray", linestyle=":", label=f"raw gap k={gap_k}")
            ax.set_xlabel("clusters before merge"); ax.set_ylabel("merge distance")
            ax.invert_xaxis(); ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        k = st.slider("k (number of clusters)", 2, 10, default_hier_k, key="hier_k")
        if st.button("Run Hierarchical", key="hier_run"):
            with st.spinner("Running Hierarchical..."):
                st.session_state["hier_result"] = run_hierarchical(non_outliers_df, k)
        if "hier_result" in st.session_state:
            result = st.session_state["hier_result"]
            st.success(f"{result['Cluster'].nunique()} clusters")
            st.plotly_chart(plot_3d_scatter(result, "Hierarchical Segments"), use_container_width=True)
            st.plotly_chart(plot_cluster_distribution(result, "Hierarchical Cluster Sizes & Feature Means"), use_container_width=True)
            feature = st.selectbox("Feature to inspect by cluster", RFM_COLUMNS, key="hier_feature")
            fig = plot_feature_distribution_by_cluster(result, feature, plot_type="violin")
            st.pyplot(fig)
            plt.close(fig)

    with sub_tabs[2]:
        st.subheader("GMM")
        if st.button("Compute parameter recommendation", key="gmm_suggest"):
            with st.spinner("Sweeping k (BIC)..."):
                rec_df = suggest_gmm_k(non_outliers_df)
                st.session_state["gmm_rec"] = rec_df
                # Pre-set before the slider below is instantiated (see DBSCAN note above) --
                # this is the fix for the recommendation not reaching "Run GMM".
                st.session_state["gmm_k"] = find_elbow_by_distance(rec_df["k"], rec_df["bic"])
        default_gmm_k = 4
        if "gmm_rec" in st.session_state:
            rec_df = st.session_state["gmm_rec"]
            # Same chord-distance elbow method used for K-Means' inertia curve, rather than
            # a fixed 5%-of-max-improvement threshold -- more robust to the exact shape of
            # the BIC curve and was landing on the wrong k on the real dataset.
            chosen_k = find_elbow_by_distance(rec_df["k"], rec_df["bic"])
            default_gmm_k = chosen_k
            st.info(f"k={chosen_k} recommended -- elbow of the BIC curve.")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(rec_df["k"], rec_df["bic"], marker="o")
            ax.axvline(chosen_k, color="red", linestyle="--")
            ax.set_xlabel("k"); ax.set_ylabel("BIC")
            st.pyplot(fig)
            plt.close(fig)
        k = st.slider("k (number of clusters)", 2, 20, default_gmm_k, key="gmm_k")
        if st.button("Run GMM", key="gmm_run"):
            with st.spinner("Running GMM..."):
                st.session_state["gmm_result"] = run_gmm(non_outliers_df, k)
        if "gmm_result" in st.session_state:
            result = st.session_state["gmm_result"]
            st.success(f"{result['Cluster'].nunique()} clusters")
            st.plotly_chart(plot_3d_scatter(result, "GMM Segments"), use_container_width=True)
            st.plotly_chart(plot_cluster_distribution(result, "GMM Cluster Sizes & Feature Means"), use_container_width=True)
            feature = st.selectbox("Feature to inspect by cluster", RFM_COLUMNS, key="gmm_feature")
            fig = plot_feature_distribution_by_cluster(result, feature, plot_type="violin")
            st.pyplot(fig)
            plt.close(fig)

    with sub_tabs[3]:
        st.subheader("Comparative Analysis")
        method_results = {"K-Means": kmeans_result}
        for name, key in [("DBSCAN", "dbscan_result"), ("Hierarchical", "hier_result"), ("GMM", "gmm_result")]:
            if key in st.session_state:
                method_results[name] = st.session_state[key]
        if len(method_results) < 2:
            st.info("Run at least one more method above to compare it against K-Means.")
        else:
            comparison_df = compare_methods(method_results).sort_values("silhouette_score", ascending=False)

            # Color the bars by score itself (red -> yellow -> green), not by hardcoding which
            # method wins -- so it stays correct if a different dataset changes the ranking.
            fig = px.bar(
                comparison_df, x="method", y="silhouette_score", title="Silhouette Score by Method",
                color="silhouette_score", color_continuous_scale="RdYlGn",
                range_color=[comparison_df["silhouette_score"].min(), comparison_df["silhouette_score"].max()],
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                comparison_df.rename(columns={
                    "method": "Method", "silhouette_score": "Silhouette Score",
                    "n_clusters": "Clusters", "n_excluded": "Excluded (noise)",
                }).style.format({"Silhouette Score": "{:.4f}"}),
                use_container_width=True, hide_index=True,
            )
            