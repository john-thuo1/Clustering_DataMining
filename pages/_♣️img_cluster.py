import os
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
import plotly.graph_objs as go
from tqdm import tqdm

TF_ENABLE_ONEDD_OPTS = '0'



@st.cache_resource()
def load_model():
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return model

model = load_model()

def load_and_extract_features(input_dir, cache_file='feature_cache.npy'):
    vgg16_features = []
    file_info = []
    feature_cache = load_feature_cache(cache_file)

    image_list = [image_name for image_name in os.listdir(input_dir) if image_name.lower().endswith(('.jpg'))]

    with tqdm(total=len(image_list), desc="Processing images", unit="image", position=0, leave=True) as pbar:
        for image_name in image_list:
            image_path = os.path.join(input_dir, image_name)

            # Check if features are already cached
            if image_path in feature_cache:
                features = feature_cache[image_path]
            else:
                try:
                    img = image.load_img(image_path, target_size=(224, 224))

                    # Convert the image to numpy array and extract VGG16 features
                    features = extract_vgg16_features(img)

                    # Update the feature cache
                    feature_cache[image_path] = features
                except Exception as e:
                    print(f'Issue with image {image_path}: {str(e)}')
                    features = None

            # Store VGG16 features in one list
            vgg16_features.append(features)

            # Store file information in another list
            file_info.append({
                'file_path': image_path,
                'file_name': image_name
            })

            pbar.update(1)

    save_feature_cache(feature_cache, cache_file)
    return vgg16_features, file_info

def extract_vgg16_features(img):
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

def load_feature_cache(cache_file):
    try:
        return np.load(cache_file, allow_pickle=True).item()
    except FileNotFoundError:
        return {}

def save_feature_cache(feature_cache, cache_file):
    np.save(cache_file, feature_cache)

def main():
    input_dir = './Data/Downloaded_images/'
    vgg16_features, file_info = load_and_extract_features(input_dir)

    optimal_clusters = 4
    matrix_pca, cluster_labels, centroids = perform_clustering(vgg16_features, n_clusters=optimal_clusters)

    # Streamlit App
    st.title('Image Clustering with Streamlit')
    st.text("""The dataset consists of different Product Images(Bicycles, Electronics, Pamphlets, Tractors) scraped from iStockPhoto LP.You can find the corresponding scraping files in the notebook.""")


    for cluster in range(optimal_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        sample_indices = cluster_indices[:3]

        st.text(f"Cluster {cluster} Images/")

        
        # Create columns for each image
        columns = st.columns(len(sample_indices))
        
        for i, idx in enumerate(sample_indices):
            # Retrieve file information based on the index
            file_data = file_info[idx]
            image_path = file_data['file_path']
            
            # Display image in the column with some spacing
            columns[i].image(image_path, caption=f'Cluster {cluster}, Image {idx}', use_column_width=True)
            
    # Scatter plot for clustered data
    fig = plot_scatter_and_centroids(matrix_pca, cluster_labels, centroids)

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
