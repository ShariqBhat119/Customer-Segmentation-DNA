# 🧬 Customer DNA

**RFM analysis + machine-learning customer segmentation, turned into a business-ready Streamlit app.**

Most retailers treat every customer the same way — no distinction between a top-spending regular and someone who bought once and never came back. Customer DNA fixes that: upload a transaction dataset, and it runs a full RFM (Recency, Frequency, Monetary) analysis, clusters customers with K-Means, and translates the output into plain-language segments and next-best-actions a business team can act on immediately — no data-science background required.

> 🔗 **Live app:**[customer-dna.streamlit.app](https://customer-dna.streamlit.app/)

---

## What it does

Customer DNA takes raw transaction-level data and turns it into a live segmentation dashboard:

- **Data Quality** — a funnel view of raw rows → cleaned rows → core customers, plus the cleaning and outlier rules applied, so the pipeline is auditable at a glance.
- **Customer Metrics** — Recency, Frequency, and Monetary Value distributions, with before/after views showing exactly what outlier removal changed.
- **Segments** — an interactive 3D view of the customer base, colored by segment, built on the same scaled RFM values K-Means actually clusters on.
- **Strategies** — each segment (REWARD, RETAIN, NURTURE, RE-ENGAGE, PAMPER, UPSELL, DELIGHT) gets its customer share, revenue share, and concrete recommended actions — including a live Pareto-style insight (e.g. *"12% of customers drive 45% of revenue"*).
- **Find a Customer** — look up any Customer ID and instantly see which segment they belong to.
- **Analytics** — a technical deep-dive for methodology review: DBSCAN, Hierarchical, and GMM clustering as alternatives to K-Means, each with its own parameter-recommendation logic, plus a side-by-side silhouette-score comparison across all four methods.

## How it works

1. **Clean** — transaction rows are validated against the Online Retail II conventions (6-digit invoice numbers, 5-digit stock codes, positive prices, no missing Customer IDs).
2. **Engineer RFM features** — each customer is reduced to Recency (days since last order), Frequency (distinct invoices), and Monetary Value (total spend).
3. **Remove outliers** — IQR-based outlier detection on Frequency and Monetary Value only (Recency is deliberately excluded — being inactive isn't a statistical outlier the way extreme spend or order count is).
4. **Segment** — K-Means (the primary, business-facing method) clusters the core customer base; cluster profiles are automatically mapped to business labels (REWARD, RETAIN, NURTURE, RE-ENGAGE) based on relative spend/frequency/recency, not assumed cluster order. Outlier customers get their own labels (PAMPER, UPSELL, DELIGHT).
5. **Validate** — DBSCAN, Hierarchical, and Gaussian Mixture Models are run as alternative methods and compared against K-Means on silhouette score, in the Analytics tab.

## Tech stack

- **App framework:** [Streamlit](https://streamlit.io)
- **Data processing:** pandas, numpy
- **Machine learning:** scikit-learn (K-Means, DBSCAN, Gaussian Mixture), scipy (hierarchical clustering)
- **Visualization:** Plotly (interactive charts), Matplotlib (static distribution plots)

## Dataset

Built for the [Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI Machine Learning Repository), and any dataset following the same conventions. Your file needs these columns:

| Column | Description |
|---|---|
| `Invoice` | Invoice number (6 digits) |
| `StockCode` | Product code (5 digits, optionally + letters) |
| `Customer ID` | Unique customer identifier |
| `InvoiceDate` | Date/time of the transaction |
| `Quantity` | Units purchased |
| `Price` | Unit price |

Accepted formats: `.xlsx`, `.xls`, `.csv` — uploaded directly through the app's sidebar, no setup required.

## Getting started

### Prerequisites

- Python 3.10+
- Git

### Setup

```bash
# Clone the repo
git clone https://github.com/hanangulzar11-max/customer-dna.git
cd customer-dna

# Create and activate a virtual environment
python -m venv venv
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app opens at https://customer-dna.streamlit.app/. Upload a dataset in the sidebar to get started.

## Project structure

```
customer-dna/
├── app.py                   # Main Streamlit app — pipeline, clustering, and all UI tabs
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml          # App theme (colors, fonts, borders)
└── README.md
```

## Customizing the theme

The app's look (colors, font, border radius, sidebar styling) is controlled entirely by `.streamlit/config.toml`. Edit the `[theme]` and `[theme.sidebar]` values and restart the app to see changes — see the [Streamlit theming docs](https://docs.streamlit.io/develop/concepts/configuration/theming) for the full set of options.

## Deployment

This app is designed to deploy directly to [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub (public repo required for the free tier).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and click **New app**.
3. Point it at this repository, branch `main`, and file `app.py`.
4. Deploy — Streamlit Cloud installs `requirements.txt` and picks up `.streamlit/config.toml` automatically.

## Team

Built as a capstone data analysis project applying the DIG Framework (Description, Introspection, Goal Setting) to the Online Retail II dataset.

- **Hanan** — feature engineering, visualization, pipeline design, and this app
- **Shariq, Tehleel, Saher** — team collaborators on the underlying analysis
- **Suha Vaqar Khan** — mentor