### Project Title: SegmentSense: Smart Banking Personalization

**Overview:**
SegmentSense is a comprehensive data analysis and machine learning project aimed at enhancing personalized banking experiences. By leveraging advanced clustering techniques and dimensionality reduction, SegmentSense segments customers based on their financial behaviors, helping banks tailor their services more effectively.

**Objectives:**
- To analyze and understand customer behavior through extensive data exploration and visualization.
- To identify distinct customer segments using various clustering algorithms.
- To ensure robust and accurate clustering by comparing multiple models and methodologies.
- To leverage dimensionality reduction for efficient data processing and visualization.
- To provide actionable insights for personalized banking services.

**Tools and Libraries:**
- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- TensorFlow/Keras (for autoencoder)
- IsolationForest (for outlier detection)

**Steps and Methodologies:**

1. **Data Exploration and Cleaning:**
    - Loaded the customer dataset and set the customer ID as the index.
    - Performed data summary and identified missing values, duplicates, and outliers.
    - Visualized the distribution of key features using histograms, boxplots, and scatter plots.
    - Handled missing values by filling or removing them appropriately.
    - Removed outliers using the Isolation Forest method to ensure data quality.

2. **Data Normalization:**
    - Applied Z-score normalization using StandardScaler to standardize the features, making them suitable for clustering.

3. **Exploratory Data Analysis (EDA):**
    - Conducted comprehensive EDA to understand the relationships and distributions of features.
    - Created various plots, including density plots, violin plots, scatter plots, and heatmaps, to visualize correlations and patterns.

4. **Dimensionality Reduction with PCA:**
    - Performed Principal Component Analysis (PCA) to reduce dimensionality and identify key features contributing to the variance.
    - Visualized the explained variance to determine the optimal number of components.
    - Generated a heatmap of component loadings to interpret the contribution of each feature.

5. **Clustering Analysis:**
    - **K-Means Clustering:**
        - Conducted K-Means clustering with a range of clusters and evaluated the optimal number using the Elbow Method and Silhouette Score.
        - Visualized the clusters in the PCA-reduced 2D space.
    - **Hierarchical Clustering:**
        - Applied Agglomerative Clustering with different linkage methods (single, complete, ward) to identify hierarchical structures.
        - Visualized dendrograms to understand the clustering hierarchy and determine the number of clusters.

6. **Dimensionality Reduction with Autoencoder:**
    - Implemented an autoencoder neural network to further reduce dimensionality and capture important features.
    - Trained the autoencoder on the scaled data and used the encoder part for dimensionality reduction.
    - Re-evaluated clustering on the reduced dimensions to ensure consistency and accuracy.

7. **Comparison and Discrepancy Check:**
    - Compared clustering results from K-Means and Hierarchical Clustering to check for discrepancies.
    - Ensured the robustness of the clusters by cross-validating with different methodologies and dimensionality reduction techniques.

**Key Insights:**
- Identified distinct customer segments based on financial behaviors, aiding in personalized service offerings.
- Visualized customer distributions and relationships between key features, providing actionable insights for banking strategies.
- Ensured data quality through outlier removal and normalization, leading to reliable clustering results.
- Leveraged advanced machine learning techniques to reduce dimensionality and enhance clustering accuracy.

**Conclusion:**
SegmentSense demonstrates a comprehensive approach to customer segmentation in the banking sector, utilizing advanced data analysis and machine learning techniques. By ensuring robust and accurate clustering, this project provides valuable insights for personalized banking services, showcasing expertise in data handling, EDA, clustering, and machine learning.

**Skills Demonstrated:**
- Data Exploration and Visualization
- Data Cleaning and Preprocessing
- Dimensionality Reduction (PCA, Autoencoder)
- Clustering Algorithms (K-Means, Hierarchical Clustering)
- Machine Learning Model Evaluation and Comparison
- Advanced Python Programming
- Use of Machine Learning Libraries (Scikit-learn, TensorFlow/Keras)


This Project was Made by :
Tushar Kanda | 221AI042 | tusharkanda.221ai042@nitk.edu.in
Priyanshu Maniyar | 221AI023 | priyanshumaniyar.221ai023@nitk.edu.in
Sachin Choudhary | 221AI034 | sachinchoudhary.221ai034@nitk.edu.in

