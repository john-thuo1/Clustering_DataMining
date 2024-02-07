import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st 
import spacy

st.set_page_config(
    page_icon="📰"
)

@st.cache_resource()
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource()
def load_data():
    df = pd.read_csv("./Data/news_data.csv", index_col="Index")
    df_news = df.copy(deep=True)
    return df, df_news

def preprocess_text(text, nlp):
    return ' '.join([token.lemma_ for token in nlp(text.lower()) if token.is_alpha and token.text not in nlp.Defaults.stop_words])


def cluster_news(df_news, vectorizer, pca, kmeans):
    df_news['Title'] = df_news['Title'].apply(lambda x: preprocess_text(x, nlp))
    tfidf_matrix = vectorizer.fit_transform(df_news['Title'])
    tfidf_matrix_pca = pca.fit_transform(tfidf_matrix.toarray())
    df_news['Cluster'] = kmeans.fit_predict(tfidf_matrix)

    df_results = df_news.reset_index()[['Index', 'Link', 'Cluster']].copy(deep=True)
    df_results_grouped = df_results.sort_values(by='Cluster').set_index('Index')

    

    return df_results_grouped, tfidf_matrix_pca

# Display the cluster Graph

def plot_clusters(tfidf_matrix_pca):
    df_plotly = pd.DataFrame({
        'PC1': tfidf_matrix_pca[:, 0],
        'PC2': tfidf_matrix_pca[:, 1],
        'Cluster': df_news['Cluster'].astype(int).astype(str)  
    })

    # Create scatterplot using Plotly Graph Objects
    fig = go.Figure()

    # Create scatter plot for all clusters in the same plot
    for cluster in sorted(df_news['Cluster'].unique()):
        cluster_data = df_plotly[df_plotly['Cluster'] == str(cluster)]
        fig.add_trace(
            go.Scatter(
                x=cluster_data['PC1'],
                y=cluster_data['PC2'],
                mode='markers',
                marker=dict(size=10),
                name=f'Cluster {cluster}'
            )
        )

    # Update layout
    fig.update_layout(
        title='Scatterplot of News Articles with Clusters',
        xaxis=dict(title='Principal Component 1'),
        yaxis=dict(title='Principal Component 2'),
    )

    return fig
# Load spacy model
nlp = load_spacy_model()

# Load Data
df, df_news = load_data()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')

# PCA for dimensionality reduction
pca = PCA(n_components=2)

# K-Means Clustering
optimal_clusters = 3  # Used Elbow method to get this.
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)



st.sidebar.success("Select a demo above.")

st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background: linear-gradient(45deg, #4E2A84, #964B00);
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content .stButton {
        color: #4E2A84;
        background-color: #FFD700;
        border: 2px solid #FFD700;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



st.title("News Clustering ✌️")
st.text("""The dataset consists of Titles and Links(urls) of various News Articles scraped from Google Search News""")

st.write("**Original News Articles Dataset**")
st.write(df.head())

st.write("**Clustered News Articles Dataset**")
st.text("""Scroll through the dataframe to look at all news with their clusters""")


# Cluster News
df_results_grouped, tfidf_matrix_pca= cluster_news(df_news, vectorizer, pca, kmeans)

# Display the clustered results
st.dataframe(df_results_grouped)



fig = plot_clusters(tfidf_matrix_pca)

st.plotly_chart(fig)


