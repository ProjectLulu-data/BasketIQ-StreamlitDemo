"""Data-integrity checks for saved BasketIQ outputs; never trains mining models."""
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED = [
    'data/clean/cleaning_summary.csv', 'data/clean/customer_features.csv', 'results/01_EDA/dashboard_kpis.csv',
    'results/02_Association_Rules/association_rules.csv', 'results/02_Association_Rules/parameter_study.csv',
    'results/03_DBSCAN/dbscan_parameter_results.csv', 'results/03_DBSCAN/dbscan_segment_summary.csv',
    'results/03_DBSCAN/dbscan_final_clusters.csv', 'results/04_PCA_KMeans/customer_cluster_comparison.csv',
    'results/04_PCA_KMeans/common_customer_silhouette_scores.csv', 'results/04_PCA_KMeans/pca_explained_variance.csv',
    'results/04_PCA_KMeans/segment_comparison_summary.csv',
]
IMAGES = [
    'results/01_EDA/monthly_revenue.png', 'results/01_EDA/top_countries.png', 'results/01_EDA/top_products.png',
    'results/01_EDA/customer_frequency.png', 'results/02_Association_Rules/rules_scatter.png', 'results/02_Association_Rules/top_rules.png',
    'results/03_DBSCAN/dbscan_clusters_pca.png', 'results/04_PCA_KMeans/customer_pca.png',
    'results/04_PCA_KMeans/kmeans_vs_dbscan_pca.png',
]

def validate_outputs(base_dir: Path) -> pd.DataFrame:
    """Return evidence-based PASS/FAIL/WARN rows; missing inputs never masquerade as success."""
    base_dir = Path(base_dir)
    checks=[]

    def log(area, name, passed, detail, warn=False):
        checks.append(dict(Area=area, Check=name, Status=('WARN' if warn else ('PASS' if passed else 'FAIL')), Details=str(detail)))

    missing=[p for p in REQUIRED if not (base_dir/p).is_file()]
    log('Files','Required CSV files present',not missing,
        'All 12 found' if not missing else 'Missing: '+', '.join(missing))
    missing_images=[p for p in IMAGES if not (base_dir/p).is_file()]
    log('Files','Charts exist',not missing_images,
        'All 9 figures found' if not missing_images else 'Missing: '+', '.join(missing_images))
    if missing:
        return pd.DataFrame(checks)

    try:
        c=pd.read_csv(base_dir/'data/clean/customer_features.csv',dtype={'CustomerID':str})
        k=pd.read_csv(base_dir/'results/01_EDA/dashboard_kpis.csv').iloc[0]
        summary=pd.read_csv(base_dir/'data/clean/cleaning_summary.csv')
        rules=pd.read_csv(base_dir/'results/02_Association_Rules/association_rules.csv')
        study=pd.read_csv(base_dir/'results/02_Association_Rules/parameter_study.csv')
        d=pd.read_csv(base_dir/'results/03_DBSCAN/dbscan_final_clusters.csv',dtype={'CustomerID':str})
        seg=pd.read_csv(base_dir/'results/03_DBSCAN/dbscan_segment_summary.csv')
        param=pd.read_csv(base_dir/'results/03_DBSCAN/dbscan_parameter_results.csv')
        comp=pd.read_csv(base_dir/'results/04_PCA_KMeans/customer_cluster_comparison.csv',dtype={'CustomerID':str})
        sil=pd.read_csv(base_dir/'results/04_PCA_KMeans/common_customer_silhouette_scores.csv')
        pca=pd.read_csv(base_dir/'results/04_PCA_KMeans/pca_explained_variance.csv')
        comp_summary=pd.read_csv(base_dir/'results/04_PCA_KMeans/segment_comparison_summary.csv')
    except (ValueError, KeyError, OSError, pd.errors.ParserError) as e:
        log('Files','Parse saved outputs',False,f'{type(e).__name__}: {e}')
        return pd.DataFrame(checks)

    try:
        counts=summary.set_index('RecordStatus')['RowCount']
        clean=int(counts['Positive-sale candidate'])
        exception=int(summary['RowCount'].sum()-clean)
        valid=(int(summary['RowCount'].sum())==1067371 and clean==1007913 and exception==59458)
        log('M1','Cleaning summary reconciles',valid,
            f'Raw {int(summary.RowCount.sum()):,}; positive {clean:,}; exceptions {exception:,}')
        valid=(int(k['merchandise_lines'])==1003357 and int(k['product_baskets'])==39516 and int(k['distinct_stockcodes'])==4895 and int(k['countries'])==43)
        log('M1','Merchandise KPI summary',valid,
            f"Lines {int(k['merchandise_lines']):,}; baskets {int(k['product_baskets']):,}; products {int(k['distinct_stockcodes']):,}; countries {int(k['countries']):,}")
        log('M1','Gross merchandise revenue',np.isfinite(k['gross_positive_sales_gbp']) and np.isclose(float(k['gross_positive_sales_gbp']),19643861.62,atol=0.01),
            f"£{float(k['gross_positive_sales_gbp']):,.2f} (before cancellations)")
        unique=c['CustomerID'].notna().all() and c['CustomerID'].is_unique
        log('M1','Customer IDs unique and present',unique,f'{len(c):,} customer profiles')
        req=['Recency','Frequency','GrossMonetary']
        valid=set(req).issubset(c.columns) and c[req].notna().all().all() and np.isfinite(c[req].to_numpy(dtype=float)).all() and (c[req]>=0).all().all()
        log('M1','RFM fields nonnegative and finite',valid,'Recency, Frequency, GrossMonetary')
        log('M1','Customer count agrees with summary',len(c)==int(k['identified_customers'])==5852,f'{len(c):,} rows')
    except (KeyError,ValueError,TypeError) as e:
        log('M1','Validate customer and cleaning data',False,f'{type(e).__name__}: {e}')

    try:
        metrics=rules[['support','confidence','lift']].apply(pd.to_numeric,errors='coerce')
        valid=metrics.notna().all().all() and metrics['support'].between(0,1,inclusive='both').all() and metrics['confidence'].between(0,1,inclusive='both').all() and (metrics.lift>1).all()
        log('M2','Rule measures valid',valid,f'{len(rules):,} rule rows; 0–1 support/confidence, lift > 1')
        chosen=study.loc[np.isclose(study['min_support'],0.01)&np.isclose(study['min_confidence'],0.5)]
        valid=(len(chosen)==1 and int(chosen.iloc[0]['rules'])==len(rules)==452 and int(chosen.iloc[0]['itemsets'])==1342)
        log('M2','Export matches selected thresholds',valid,'Support 0.01; confidence 0.5; 1,342 itemsets; 452 rules')
        log('M2','12-configuration support/confidence study',len(study)==12,f'{len(study)} settings')
    except (KeyError,ValueError,TypeError) as e:
        log('M2','Validate association results',False,f'{type(e).__name__}: {e}')

    try:
        id_d=set(d['CustomerID']); id_c=set(c['CustomerID']); id_comp=set(comp['CustomerID'])
        valid=len(d)==len(c)==len(comp) and id_c==id_d==id_comp and d['CustomerID'].is_unique and comp['CustomerID'].is_unique
        log('M3/M4','All CustomerID sets match',valid,
            f'M1={len(c):,}; M3={len(d):,}; M4={len(comp):,}')
        observed=d['DBSCAN_Cluster'].value_counts().to_dict()
        expected=seg.set_index('DBSCAN_Cluster')['Customers'].to_dict()
        valid=observed==expected and sum(observed.values())==5852
        log('M3','DBSCAN segment summary matches labels',valid,
            'Observed '+str({int(key):int(val) for key,val in observed.items()}))
        log('M3','18 DBSCAN parameter combinations',len(param)==18,f'{len(param)} settings')
        sel=param.loc[np.isclose(param['eps'],0.4)&(param['min_samples']==15)]
        valid=len(sel)==1 and int(sel.iloc[0]['n_clusters'])==2 and int(sel.iloc[0]['n_noise'])==169 and np.isclose(float(sel.iloc[0]['silhouette']),.3212,atol=.0001)
        log('M3','Selected DBSCAN parameters and score',valid,'eps=0.4; min_samples=15; 169 noise; silhouette≈0.3212')
    except (KeyError,ValueError,TypeError) as e:
        log('M3','Validate DBSCAN outputs',False,f'{type(e).__name__}: {e}')

    try:
        log('M4','All CI3237 K-Means labels populated',comp['BDA_KMeans_Cluster'].notna().all(),
            f'{int(comp.BDA_KMeans_Cluster.notna().sum()):,}/{len(comp):,} labels')
        non_noise=comp.loc[comp['DBSCAN_Cluster']!=-1]
        valid=(len(non_noise)==5683 and len(sil)==2 and (sil['Customers_evaluated']==5683).all())
        log('M4','Same-subset silhouette sample size',valid,f'{len(non_noise):,} non-noise customers')
        vals={str(x['Method']):float(x['Silhouette']) for _,x in sil.iterrows()}
        valid=len(vals)==2 and any('K-Means' in key and np.isclose(val,.405327,atol=.00001) for key,val in vals.items()) and any('DBSCAN' in key and np.isclose(val,.321219,atol=.00001) for key,val in vals.items())
        log('M4','Saved model-comparison scores',valid,str({k:round(v,6) for k,v in vals.items()}))
        variance=pca['Explained_variance_ratio'].sum()
        log('M4','PCA two-component variance',len(pca)==2 and np.isclose(variance,.950312,atol=.0001),f'{variance:.2%}')
        valid=(comp_summary.groupby('Method')['Customers'].sum()==5852).all()
        log('M4','Comparison segment counts',valid,'Every method summarises all 5,852 customers')
    except (KeyError,ValueError,TypeError) as e:
        log('M4','Validate PCA/K-Means comparison',False,f'{type(e).__name__}: {e}')

    return pd.DataFrame(checks)
