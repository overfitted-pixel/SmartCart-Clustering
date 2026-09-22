SmartCart Clustering System

Customer segmentation for an e-commerce platform using unsupervised machine learning. The system groups customers into behaviourally meaningful clusters based on demographics, spending, and engagement — replacing SmartCart's one-size-fits-all marketing with data-driven, targeted strategies.

Problem Statement

SmartCart currently uses generic marketing and engagement strategies for all customers, with no clear understanding of behaviour patterns. This leads to inefficient marketing spend, missed opportunities to retain high-value customers, and delayed identification of churn-prone users.

This project builds an intelligent customer segmentation system that analyses historical transaction data (2,240 customers, 22 attributes) and clusters customers by purchasing behaviour, engagement level, and loyalty indicators to support personalised marketing and retention decisions.

Dataset

Each row represents one customer.

Category	Features
Demographics	Year_Birth, Education, Marital_Status, Income, Kidhome, Teenhome, Dt_Customer
Spending	MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds
Purchase Frequency	NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth
Feedback / Other	Recency, Complain, Response
Pipeline

1. Data Cleaning

Filled 24 missing Income values with the median
Dropped rows with unrealistic outliers (Age > 90, Income > 600,000) — 2,236 customers retained

2. Feature Engineering

Age — derived from Year_Birth
Customer_Tenure_Days — days since enrollment, relative to the most recent signup
Total_Spending — sum of all six spending categories
Total_Children — Kidhome + Teenhome
Education simplified into 3 tiers: Undergraduate, Graduate, Postgraduate
Living_With derived from Marital_Status: Partner vs. Alone
Raw ID, birth year, marital status, and per-category spending columns dropped after engineering

3. Encoding & Scaling

One-hot encoding for Education and Living_With
StandardScaler applied to all features

4. Dimensionality Reduction

PCA reduced to 3 components (~45% cumulative explained variance) for clustering and visualization

5. Determining Cluster Count

Elbow method (WCSS) and Silhouette Score both pointed to k = 4 as optimal

6. Clustering

K-Means (k=4) and Agglomerative Clustering (ward linkage, k=4) were both fit on the PCA-reduced data
Agglomerative Clustering labels were used for final cluster characterization
Cluster Profiles
Cluster	Income	Total Spending	Total Children	Living Situation	Response Rate	Profile
0	~$39.7K	~$222	~1.24	Partnered	7.6%	Budget-conscious families
1	~$72.8K	~$1,237	~0.51	Partnered	16.7%	High-value couples
2	~$37.0K	~$166	~1.27	Living alone	14.2%	Budget-conscious singles
3	~$70.7K	~$1,190	~0.46	Living alone	32.0%	Premium, high-response singles

Key patterns observed:

Spending scales strongly with income — the two high-income clusters (1, 3) spend ~5–7x more than the two low-income clusters (0, 2)
Higher-income clusters buy more through catalog and in-store channels, while lower-income clusters rely more on web visits and deal purchases
Household composition splits cleanly along Living_With, independent of income — one high/low-income pair is partnered, the other lives alone
Cluster 3 (high-income, living alone) is by far the most responsive to marketing campaigns, making it the strongest target for premium/personalised offers
Cluster 2 (lower-income, living alone, kids) is the least engaged and lowest-spending group, and a candidate for retention-focused, value-driven campaigns
Tech Stack
Python — pandas, NumPy
scikit-learn — StandardScaler, OneHotEncoder, PCA, KMeans, AgglomerativeClustering, silhouette_score
kneed — automated elbow-point detection
matplotlib / seaborn — EDA, correlation heatmap, PCA visualization, cluster plots
Project Structure
smartcart-clustering/
├── smartcart_customers.csv       # raw dataset
├── smartcart.ipynb               # full analysis notebook
└── README.md
How to Run
bash
pip install pandas numpy scikit-learn kneed matplotlib seaborn
jupyter notebook smartcart.ipynb
Business Impact

These segments let SmartCart move from generic campaigns to targeted strategies: premium offers for high-value singles (Cluster 3), loyalty/upsell campaigns for high-value couples (Cluster 1), and value-focused retention campaigns for the two budget-conscious segments (Clusters 0 and 2) most at risk of disengagement.
