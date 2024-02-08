import os
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
import plotly.graph_objs as go

@st.cache_resource()
def load_model():
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return model

def load_and_extract_features(input_dir):
    vgg16_features = []
    file_info = []

    for image_name in os.listdir(input_dir):
        if image_name.lower().endswith(('.jpg')):
            image_path = os.path.join(input_dir, image_name)
            try:
                img = image.load_img(image_path, target_size=(224, 224))

                # Convert the image to numpy array and extract VGG16 features
                features = extract_vgg16_features(img)

                # Store VGG16 features in one list
                vgg16_features.append(features)

                # Store file information in another list
                file_info.append({
                    'file_path': image_path,
                    'file_name': image_name
                })
            except Exception as e:
                print(f'Issue with image {image_path}: {str(e)}')

    return vgg16_features, file_info

def extract_vgg16_features(img):

    model = load_model()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features = features.flatten()
    features = features / np.linalg.norm(features)  # Normalized features
    return features

def perform_clustering(vgg16_features, n_clusters=4):
    pca = PCA(n_components=2)
    matrix_pca = pca.fit_transform(vgg16_features)

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(matrix_pca)
    centroids = kmeans.cluster_centers_

    return matrix_pca, cluster_labels, centroids

def plot_scatter_and_centroids(matrix_pca, cluster_labels, centroids):
    cluster_colors = ['yellow', 'green', 'blue', 'purple']
    scatter_traces = []
    unique_clusters = np.unique(cluster_labels)

    for i, cluster in enumerate(unique_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_trace = go.Scatter(
            x=matrix_pca[cluster_indices, 0],
            y=matrix_pca[cluster_indices, 1],
            mode='markers',
            marker=dict(
                color=cluster_colors[i],
                size=8,
                line=dict(color='black', width=1)
            ),
            name=f'Cluster {cluster}'
        )
        scatter_traces.append(cluster_trace)

    centroid_trace = go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode='markers',
        marker=dict(
            color='red',
            size=14,
            symbol='x'
        ),
        name='Centroids'
    )

    layout = go.Layout(
        title='Scatter Plot with Clustered Data and Centroids',
        xaxis=dict(title='Principal Component 1'),
        yaxis=dict(title='Principal Component 2'),
    )

    fig = go.Figure(data=scatter_traces + [centroid_trace], layout=layout)

    return fig

def main():
    input_dir = './Data/Downloaded_images/'
    vgg16_features, file_info = load_and_extract_features(input_dir)

    optimal_clusters = 4
    matrix_pca, cluster_labels, centroids = perform_clustering(vgg16_features, n_clusters=optimal_clusters)

    # Streamlit App
    st.title('Image Clustering with Streamlit')

    # Scatter plot for clustered data
    fig = plot_scatter_and_centroids(matrix_pca, cluster_labels, centroids)

    # Show the scatter plot using Streamlit
    st.plotly_chart(fig)

    # Visualize 3 images for each cluster
    for cluster in range(optimal_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        sample_indices = cluster_indices[:3]

        st.subheader(f'Cluster {cluster} Images:')
        st.image([file_info[idx]['file_path'] for idx in sample_indices], width=200)

if __name__ == "__main__":
    main()
