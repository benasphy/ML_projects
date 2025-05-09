import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import io as io_lib

def compress_image(image, n_components):
    # Reshape image to 2D array
    h, w, d = image.shape
    image_2d = image.reshape(h * w, d)
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, d))
    compressed = pca.fit_transform(image_2d)
    
    # Reconstruct image
    reconstructed = pca.inverse_transform(compressed)
    reconstructed = reconstructed.reshape(h, w, d)
    
    # Clip values to valid range
    reconstructed = np.clip(reconstructed, 0, 1)
    
    # Calculate actual compressed size
    # For each pixel, we store:
    # 1. n_components values (compressed data)
    # 2. mean vector (d values)
    # 3. component vectors (n_components * d values)
    compressed_size = (h * w * n_components +  # compressed data
                      d +                      # mean vector
                      n_components * d)        # component vectors
    
    return reconstructed, pca.explained_variance_ratio_, compressed_size

def run():
    st.header("Image Compression using PCA")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/Dimensionality_Reduction)", unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Read image
        image = io.imread(uploaded_file)
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
        # Convert to float and normalize
        image_float = image.astype(np.float32) / 255.0
        
        # Parameters
        st.subheader("Compression Parameters")
        max_components = min(image.shape[2], 3)  # Limit to number of color channels
        n_components = st.slider("Number of Components", min_value=1, max_value=max_components, value=1)
        
        if st.button("Compress Image"):
            # Compress image
            compressed_image, explained_variance, compressed_size = compress_image(image_float, n_components)
            
            # Convert back to uint8 for display
            compressed_display = (compressed_image * 255).astype(np.uint8)
            
            # Display compressed image
            st.subheader("Compressed Image")
            st.image(compressed_display, use_column_width=True)
            
            # Display compression statistics
            st.subheader("Compression Statistics")
            original_size = image.size  # Number of pixels * number of channels
            compression_ratio = original_size / compressed_size
            st.write(f"Original Size: {original_size/1024:.1f} KB")
            st.write(f"Compressed Size: {compressed_size/1024:.1f} KB")
            st.write(f"Compression Ratio: {compression_ratio:.1f}x")
            st.write(f"Components Used: {n_components}")
            st.write(f"Explained Variance: {np.sum(explained_variance)*100:.1f}%")
            
            # Display explained variance
            st.subheader("Explained Variance")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(1, len(explained_variance) + 1)),
                y=explained_variance,
                name='Individual'
            ))
            fig.add_trace(go.Scatter(
                x=list(range(1, len(explained_variance) + 1)),
                y=np.cumsum(explained_variance),
                name='Cumulative',
                mode='lines+markers'
            ))
            fig.update_layout(
                title='Explained Variance by Component',
                xaxis_title='Component',
                yaxis_title='Explained Variance',
                showlegend=True
            )
            st.plotly_chart(fig)
            
            # Download compressed image
            compressed_pil = Image.fromarray(compressed_display)
            img_byte_arr = io_lib.BytesIO()
            compressed_pil.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            st.download_button(
                label="Download Compressed Image",
                data=img_byte_arr,
                file_name="compressed_image.png",
                mime="image/png"
            )

if __name__ == "__main__":
    run() 