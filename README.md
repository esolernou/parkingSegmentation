# parkingSegmentation
parking users segmentation from the parking system operational log

1. Preprocess.py 
  Adds new variables to original log dataset
  Input file: history_origin.csv
  Output file: history_2022_02_preprocessed_.csv
  
2. Exploratory_history_22_02.py
  Cleans the data and creates Users dataset
  Input file: history_2022_02_preprocessed_.csv
  Output files: users_2022_02.csv  +  history_2022_02_to_powerbi.csv
  
3. Exploratory_users_22_02.py
   Cleans the data and does some exploratory analysis
   Overwrites the file users_2022_02.csv 
   Input file:  users_2022_02.csv 

4. scale_and_pca.py
   Scales data and applies Principal Component Analysis
   Input file:  users_2022_02.csv
   Output file: explained_var.csv
   
5. kmeans.py
   Clustering with KMeans from sklearn.cluster package 
   Input file: users_feb_pca.csv
   Output files: users_2022_2_labeled.csv + centers_kmeans6.csv
   
6. hierarchical.py
   Clustering with AgglomerativeClustering from sklearn.cluster package
   Input file: users_2022_2_labeled.csv
   Output file: users_2022_2_labeled.csv
   
7. som.py
   Clustering with minisom from MiniSom package
   Input file: users_2022_2_labeled.csv
   Output files: users_2022_2_labeled.csv + centers_som7.csv



Also uploaded DBSCAN and Gaussian Mixture models code, although they don't fit the data
