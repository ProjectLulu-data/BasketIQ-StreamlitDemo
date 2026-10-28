from pathlib import Path

import pandas as pd
import streamlit as st

from quality import validate_outputs


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="BasketIQ | Retail Intelligence", page_icon="🛒", layout="wide")


@st.cache_data(show_spinner=False)
def load_table(relative_path):
    path = ROOT / relative_path
    if not path.is_file():
        st.error(f"Required data file not found: {relative_path}")
        st.stop()
    options = {"dtype": {"CustomerID": str}} if any(
        part in relative_path for part in ("customer_features", "final_clusters", "cluster_comparison")
    ) else {}
    return pd.read_csv(path, low_memory=False, **options)


def show_chart(relative_path, caption=None):
    path = ROOT / relative_path
    if not path.is_file():
        st.error(f"Chart not found: {relative_path}")
        return
    st.image(str(path), use_container_width=True)
    if caption:
        st.caption(caption)


def show_overview():
    st.header("Retail overview")
    kpis = load_table("results/01_EDA/dashboard_kpis.csv").iloc[0]
    cols = st.columns(4)
    cols[0].metric("Customers", f"{int(kpis['identified_customers']):,}")
    cols[1].metric("Purchase baskets", f"{int(kpis['product_baskets']):,}")
    cols[2].metric("Merchandise lines", f"{int(kpis['merchandise_lines']):,}")
    cols[3].metric("Gross merchandise sales", f"£{float(kpis['gross_positive_sales_gbp']) / 1_000_000:.2f}m")

    st.write(
        "Exploration of retail transaction trends, top-selling products, geographic sales "
        "and customer purchase frequency."
    )
    tabs = st.tabs(["Monthly sales", "Countries", "Products", "Customer frequency"])
    with tabs[0]:
        show_chart("results/01_EDA/monthly_revenue.png")
        st.caption("Gross positive sales before cancellations; December 2011 is a partial month.")
    with tabs[1]:
        show_chart("results/01_EDA/top_countries.png")
    with tabs[2]:
        show_chart("results/01_EDA/top_products.png")
        st.caption("Product quantities are gross positive units before returns or cancellations.")
    with tabs[3]:
        show_chart("results/01_EDA/customer_frequency.png")
    with st.expander("Dataset summary"):
        st.write(
            f"{int(kpis['distinct_stockcodes']):,} product codes across "
            f"{int(kpis['countries'])} countries."
        )
        st.dataframe(load_table("data/clean/cleaning_summary.csv"),
                     use_container_width=True, hide_index=True)


def show_associations():
    st.header("Association rule mining")
    st.write("Apriori identifies products that frequently appear together in customer baskets.")
    rules = load_table("results/02_Association_Rules/association_rules.csv").copy()
    parameter_study = load_table("results/02_Association_Rules/parameter_study.csv")
    for column in ("support", "confidence", "lift"):
        rules[column] = pd.to_numeric(rules[column], errors="coerce")

    cols = st.columns(3)
    support = cols[0].slider("Minimum support", 0.0, 0.05, 0.01, 0.001)
    confidence = cols[1].slider("Minimum confidence", 0.0, 1.0, 0.5, 0.05)
    lift = cols[2].slider("Minimum lift (exclusive)", 0.0, 55.0, 1.0, 0.5)
    selected = rules.loc[
        (rules["support"] >= support)
        & (rules["confidence"] >= confidence)
        & (rules["lift"] > lift)
    ].sort_values("lift", ascending=False)

    col1, col2 = st.columns(2)
    col1.metric("Total rules", f"{len(rules):,}")
    col2.metric("Matching rules", f"{len(selected):,}")
    st.dataframe(selected, use_container_width=True, hide_index=True)
    st.download_button(
        "Download selected rules",
        selected.to_csv(index=False).encode("utf-8"),
        file_name="BasketIQ_association_rules.csv", mime="text/csv",
    )
    with st.expander("Threshold experiments (saved runs; not affected by sliders)"):
        st.dataframe(parameter_study, use_container_width=True, hide_index=True)
    tab1, tab2 = st.tabs(["Rule distribution", "Highest-lift rules"])
    if selected.empty:
        with tab1:
            st.info("No rules match these thresholds. Try lowering a filter.")
        with tab2:
            st.info("No rules match these thresholds. Try lowering a filter.")
    else:
        with tab1:
            st.scatter_chart(selected, x="support", y="confidence", size="lift")
            st.caption("Rule support and confidence for the currently filtered rules; bubble size indicates lift.")
        with tab2:
            top_rules = selected.head(10).reset_index(drop=True).copy()
            top_rules.index = top_rules.index + 1
            st.bar_chart(top_rules["lift"])
            top_rules["Rule"] = top_rules["if_bought"] + " → " + top_rules["also_buys"]
            st.dataframe(
                top_rules[["Rule", "support", "confidence", "lift"]],
                use_container_width=True, hide_index=True,
            )
            st.caption("Bars correspond to the rule rows in the table below, ranked by lift.")
    st.caption("The rule table and both charts follow your filters. Apriori results are loaded from saved CSVs; the algorithm is not rerun.")


def show_dbscan():
    st.header("DBSCAN customer segmentation")
    st.write("Density-based clustering of customers using Recency, Frequency and GrossMonetary.")
    experiments = load_table("results/03_DBSCAN/dbscan_parameter_results.csv")
    segments = load_table("results/03_DBSCAN/dbscan_segment_summary.csv")
    customers = load_table("results/03_DBSCAN/dbscan_final_clusters.csv")
    chosen = experiments.loc[
        (experiments["eps"] == 0.4) & (experiments["min_samples"] == 15)
    ].iloc[0]

    cols = st.columns(4)
    cols[0].metric("Customers", f"{len(customers):,}")
    cols[1].metric("Clusters", f"{int(chosen['n_clusters'])}")
    cols[2].metric("Noise points", f"{int(chosen['n_noise']):,}")
    cols[3].metric("Silhouette score", f"{float(chosen['silhouette']):.4f}")
    st.caption("Selected parameters: ε = 0.4, MinPts = 15. The silhouette excludes noise points.")

    tab1, tab2 = st.tabs(["Customer clusters", "Parameter evaluation"])
    with tab1:
        show_chart("results/03_DBSCAN/dbscan_clusters_pca.png")
        st.dataframe(segments, use_container_width=True, hide_index=True)
        labels = sorted(int(value) for value in customers["DBSCAN_Cluster"].unique())
        cluster = st.selectbox(
            "View customers in", labels,
            format_func=lambda value: "Noise (-1)" if value == -1 else f"Cluster {value}",
        )
        matching = customers.loc[customers["DBSCAN_Cluster"] == cluster]
        display_columns = [
            name for name in ("CustomerID", "Recency", "Frequency", "GrossMonetary")
            if name in matching.columns
        ]
        st.dataframe(matching[display_columns].head(100),
                     use_container_width=True, hide_index=True)
        st.caption(f"Showing up to 100 of {len(matching):,} customers in this group.")
    with tab2:
        st.dataframe(experiments, use_container_width=True, hide_index=True)


def show_comparison():
    st.header("PCA and clustering comparison")
    st.write("Visualising customer features in two dimensions and comparing DBSCAN with CI3237 K-Means.")
    variance = load_table("results/04_PCA_KMeans/pca_explained_variance.csv")
    scores = load_table("results/04_PCA_KMeans/common_customer_silhouette_scores.csv")
    segments = load_table("results/04_PCA_KMeans/segment_comparison_summary.csv")

    cols = st.columns(2)
    cols[0].metric("PCA variance retained", f"{variance['Explained_variance_ratio'].sum():.2%}")
    cols[1].metric("Customers compared", f"{int(scores['Customers_evaluated'].iloc[0]):,}")
    tab1, tab2 = st.tabs(["PCA visualisation", "DBSCAN vs K-Means"])
    with tab1:
        show_chart("results/04_PCA_KMeans/customer_pca.png")
        st.dataframe(variance, use_container_width=True, hide_index=True)
    with tab2:
        show_chart("results/04_PCA_KMeans/kmeans_vs_dbscan_pca.png")
        st.subheader("Silhouette comparison")
        st.dataframe(scores, use_container_width=True, hide_index=True)
        st.subheader("Customer profiles")
        st.dataframe(segments, use_container_width=True, hide_index=True)
        st.caption(
            "CI3237 K-Means labels and DBSCAN labels are compared on the same 5,683 "
            "non-noise customers in the standardised RFM feature space."
        )


def show_validation():
    st.header("Data validation")
    st.write("Consistency checks for the datasets, rule tables, cluster assignments and charts.")
    if st.button("Run validation", type="primary"):
        st.session_state["validation"] = validate_outputs(ROOT)
    if "validation" in st.session_state:
        results = st.session_state["validation"]
        passed = int((results["Status"] == "PASS").sum())
        failed = int((results["Status"] == "FAIL").sum())
        cols = st.columns(2)
        cols[0].metric("Passed", passed)
        cols[1].metric("Failed", failed)
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button(
            "Download validation results", results.to_csv(index=False).encode("utf-8"),
            file_name="BasketIQ_validation_results.csv", mime="text/csv",
        )
        if failed:
            st.error("Some validation checks failed. Review the results above.")
        else:
            st.success("All saved-data consistency checks passed.")
    st.subheader("Cleaning summary")
    st.dataframe(load_table("data/clean/cleaning_summary.csv"),
                 use_container_width=True, hide_index=True)


PAGES = ["Overview & EDA", "Association Rules", "DBSCAN", "PCA & K-Means", "Data Validation"]
section = st.sidebar.radio("Sections", PAGES)
st.sidebar.divider()
st.sidebar.caption("CI3249 Advanced Data Mining")
st.title("BasketIQ")
st.caption("Retail market intelligence | Online Retail II")

if section == PAGES[0]:
    show_overview()
elif section == PAGES[1]:
    show_associations()
elif section == PAGES[2]:
    show_dbscan()
elif section == PAGES[3]:
    show_comparison()
else:
    show_validation()
