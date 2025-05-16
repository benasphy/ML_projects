import streamlit as st
import numpy as np
from sklearn.mixture import GaussianMixture
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import cv2
from skimage import color
import matplotlib.pyplot as plt
from collections import Counter

def calculate_color_stats(img_array):
    """Calculate color statistics for the image."""
    hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    h_mean, h_std = np.mean(hsv_img[:,:,0]), np.std(hsv_img[:,:,0])
    s_mean, s_std = np.mean(hsv_img[:,:,1]), np.std(hsv_img[:,:,1])
    v_mean, v_std = np.mean(hsv_img[:,:,2]), np.std(hsv_img[:,:,2])
    return {
        'hue': {'mean': h_mean, 'std': h_std},
        'saturation': {'mean': s_mean, 'std': s_std},
        'value': {'mean': v_mean, 'std': v_std}
    }

def create_color_palette(colors, counts):
    """Create a color palette visualization."""
    sorted_indices = np.argsort(counts)[::-1]
    colors = colors[sorted_indices]
    counts = counts[sorted_indices]
    palette_height = 100
    palette_width = 500
    palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
    total_count = np.sum(counts)
    segment_widths = (counts / total_count * palette_width).astype(int)
    start_x = 0
    for color, width in zip(colors, segment_widths):
        palette[:, start_x:start_x + width] = color
        start_x += width
    return palette

def run():
    st.header("Image Color Segmentation using GMM")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/GMM)", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        img_array = np.array(image)
        
        # Image Analysis
        st.subheader("Image Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Image Statistics:**")
            st.write(f"Dimensions: {img_array.shape[0]}x{img_array.shape[1]}")
            st.write(f"Total Pixels: {img_array.shape[0] * img_array.shape[1]}")
            
            color_stats = calculate_color_stats(img_array)
            st.write("\n**Color Statistics:**")
            st.write("Hue (mean ± std):", f"{color_stats['hue']['mean']:.1f} ± {color_stats['hue']['std']:.1f}")
            st.write("Saturation (mean ± std):", f"{color_stats['saturation']['mean']:.1f} ± {color_stats['saturation']['std']:.1f}")
            st.write("Value (mean ± std):", f"{color_stats['value']['mean']:.1f} ± {color_stats['value']['std']:.1f}")
        
        with col2:
            st.write("**Color Distribution:**")
            sample_size = min(10000, len(img_array.reshape(-1, 3)))
            sampled_pixels = img_array.reshape(-1, 3)[np.random.choice(len(img_array.reshape(-1, 3)), sample_size)]
            
            fig = px.scatter_3d(
                x=sampled_pixels[:, 0], y=sampled_pixels[:, 1], z=sampled_pixels[:, 2],
                color=sampled_pixels.mean(axis=1),
                title='Original Image Color Distribution'
            )
            fig.update_layout(scene=dict(
                xaxis_title='Red',
                yaxis_title='Green',
                zaxis_title='Blue'
            ))
            st.plotly_chart(fig)

        h, w, d = img_array.shape
        img_reshaped = img_array.reshape(-1, d)

        st.subheader("GMM Parameters")
        col1, col2 = st.columns(2)
        with col1:
            n_components = st.slider("Number of Color Clusters", min_value=2, max_value=10, value=5)
        with col2:
            covariance_type = st.selectbox("Covariance Type", ['full', 'tied', 'diag', 'spherical'])

        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
        clusters = gmm.fit_predict(img_reshaped)

        segmented_img = gmm.means_[clusters].reshape(h, w, d).astype(np.uint8)
        
        st.subheader("Segmentation Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Image**")
            st.image(image, use_column_width=True)
        
        with col2:
            st.write("**Segmented Image**")
            st.image(segmented_img, use_column_width=True)

        st.subheader("Color Analysis")
        
        unique_colors, counts = np.unique(segmented_img.reshape(-1, d), axis=0, return_counts=True)
        
        palette = create_color_palette(unique_colors, counts)
        st.write("**Color Palette:**")
        st.image(palette, use_column_width=True)
        
        fig = go.Figure()
        for i, (color, count) in enumerate(zip(unique_colors, counts)):
            fig.add_trace(go.Bar(
                x=[f'Cluster {i}'],
                y=[count],
                marker_color=f'rgb({color[0]},{color[1]},{color[2]})',
                name=f'Cluster {i}'
            ))
        
        fig.update_layout(
            title='Color Distribution in Segmented Image',
            xaxis_title='Cluster',
            yaxis_title='Number of Pixels',
            showlegend=False
        )
        st.plotly_chart(fig)

        st.subheader("Color Space Analysis")
        
        sample_size = min(10000, len(img_reshaped))
        indices = np.random.choice(len(img_reshaped), sample_size, replace=False)
        sampled_colors = img_reshaped[indices]
        sampled_clusters = clusters[indices]

        fig_3d = go.Figure(data=[go.Scatter3d(
            x=sampled_colors[:, 0],
            y=sampled_colors[:, 1],
            z=sampled_colors[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=sampled_clusters,
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig_3d.update_layout(
            title='Color Space Clustering',
            scene=dict(
                xaxis_title='Red',
                yaxis_title='Green',
                zaxis_title='Blue'
            )
        )
        st.plotly_chart(fig_3d)

        st.subheader("Model Evaluation")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BIC Score", f"{gmm.bic(img_reshaped):.2f}")
        with col2:
            st.metric("AIC Score", f"{gmm.aic(img_reshaped):.2f}")
        with col3:
            st.metric("Convergence Iterations", gmm.n_iter_)

        st.subheader("Cluster Information")
        for i in range(n_components):
            cluster_mask = clusters == i
            cluster_size = np.sum(cluster_mask)
            percentage = (cluster_size / len(clusters)) * 100
            mean_color = gmm.means_[i].astype(int)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                color_swatch = np.full((50, 50, 3), mean_color, dtype=np.uint8)
                st.image(color_swatch, width=100)
            
            with col2:
                st.write(f"**Cluster {i}:**")
                st.write(f"Size: {cluster_size} pixels ({percentage:.1f}%)")
                st.write(f"Mean Color: RGB({mean_color[0]}, {mean_color[1]}, {mean_color[2]})")
                st.write(f"Covariance Type: {covariance_type}")

    else:
        st.info("Please upload an image to begin color segmentation")

if __name__ == "__main__":
    run() 