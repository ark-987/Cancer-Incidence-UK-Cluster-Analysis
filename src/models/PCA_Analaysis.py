from sklearn.decomposition import PCA
#Reduce to 2D for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
pca_result=X_pca
print(pca_result)
