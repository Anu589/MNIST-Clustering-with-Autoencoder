### **MNIST Clustering with Autoencoder**  
*Unsupervised Learning | Deep Learning | K-Means | PCA | Hierarchical Clustering*  

## **Project Overview**  
This project applies **autoencoders** to compress **MNIST digits** into a **low-dimensional latent space** and clusters them using **K-Means, Hierarchical Clustering, and DBSCAN**.  

- **Autoencoder** for feature extraction  
- **Elbow Method** to find the optimal number of clusters  
- **K-Means, Hierarchical Clustering, and DBSCAN** for clustering  
- **PCA for 2D visualization**  
- **Local Outlier Factor (LOF) for anomaly detection**  
- **Reconstruction of images using Autoencoder**  

---

##  **Project Structure**  
```
 MNIST-Clustering-Autoencoder
│──  README.md         
│──  mnist_clustering.ipynb   
```

---
##  **Evaluation Metrics**  
-  **Adjusted Rand Index (ARI):** Measures clustering performance against ground truth.  
 - **Normalized Mutual Information (NMI):** Measures the mutual information between true and predicted clusters.  
- **Silhouette Score:** Measures how similar points are within a cluster.  

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

ari = adjusted_rand_score(y_test, kmeans_labels)
nmi = normalized_mutual_info_score(y_test, kmeans_labels)
silhouette = silhouette_score(encoded_imgs, kmeans_labels)

print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}, Silhouette Score: {silhouette:.4f}")
```

---

##  **References**
 MNIST Dataset - [LeCun et al.](http://yann.lecun.com/exdb/mnist/)  
 K-Means Clustering - [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/clustering.html#k-means)  
 Autoencoders - [Deep Learning Book by Goodfellow](https://www.deeplearningbook.org/)  

---

##  **Contributing**
 Contributions are welcome! Fork the repo, create a pull request, and let's improve this together.

---
