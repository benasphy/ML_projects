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

def fuzzy_c_means(data, n_clusters, m=2, max_iter=100, error=1e-5):
    # Initialize membership matrix randomly
    n_samples = data.shape[0]
    membership = np.random.random((n_samples, n_clusters))
    membership = membership / membership.sum(axis=1)[:, np.newaxis]
    
    # Initialize cluster centers
    centers = np.zeros((n_clusters, data.shape[1]))
    
    # Iterate until convergence
    for _ in range(max_iter):
        # Update cluster centers
        for j in range(n_clusters):
            centers[j] = np.sum(membership[:, j:j+1] ** m * data, axis=0) / np.sum(membership[:, j:j+1] ** m)
        
        # Update membership matrix
        old_membership = membership.copy()
        
        # Calculate distances between data points and centers
        distances = np.zeros((n_samples, n_clusters))
        for j in range(n_clusters):
            distances[:, j] = np.sum((data - centers[j]) ** 2, axis=1)
        
        # Update membership values
        for j in range(n_clusters):
            membership[:, j] = 1 / np.sum((distances[:, j:j+1] / distances) ** (1/(m-1)), axis=1)
        
        # Check convergence
        if np.max(np.abs(membership - old_membership)) < error:
            break
    
    return membership, centers

def run():
    st.header("Image Segmentation using Fuzzy C-Means")
    st.markdown("[View this project on GitHub](https://github.com/benasphy/ML_projects/tree/main/Fuzzy_C_Means)", unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Read image
        image = io.imread(uploaded_file)
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
        # Convert to LAB color space for better segmentation
        if len(image.shape) == 3:  # Color image
            # Convert to float and normalize
            image_float = image.astype(np.float32) / 255.0
            
            # Parameters
            st.subheader("Segmentation Parameters")
            n_segments = st.slider("Number of Segments", min_value=2, max_value=8, value=4)
            fuzziness = st.slider("Fuzziness Parameter (m)", min_value=1.1, max_value=3.0, value=2.0, step=0.1)
            
            if st.button("Segment Image"):
                # Reshape image for clustering
                h, w, d = image_float.shape
                image_2d = image_float.reshape(h * w, d)
                
                # Apply Fuzzy C-Means
                membership, centers = fuzzy_c_means(image_2d, n_segments, m=fuzziness)
                
                # Get segment labels
                labels = np.argmax(membership, axis=1)
                
                # Create segmented image
                segmented = centers[labels].reshape(h, w, d)
                segmented = np.clip(segmented * 255, 0, 255).astype(np.uint8)
                
                # Display segmented image
                st.subheader("Segmented Image")
                st.image(segmented, use_column_width=True)
                
                # Display membership maps
                st.subheader("Membership Maps")
                fig, axes = plt.subplots(1, n_segments, figsize=(15, 5))
                for i in range(n_segments):
                    membership_map = membership[:, i].reshape(h, w)
                    axes[i].imshow(membership_map, cmap='viridis')
                    axes[i].set_title(f'Segment {i+1}')
                    axes[i].axis('off')
                st.pyplot(fig)
                
                # Display segment statistics
                st.subheader("Segment Statistics")
                for i in range(n_segments):
                    segment_size = np.sum(labels == i)
                    segment_percentage = (segment_size / (h * w)) * 100
                    st.write(f"Segment {i+1}: {segment_size} pixels ({segment_percentage:.1f}%)")
                
                # Display segment colors
                st.subheader("Segment Colors")
                fig, ax = plt.subplots(figsize=(10, 2))
                for i in range(n_segments):
                    color = centers[i]
                    # Convert color to RGB tuple for matplotlib
                    rgb_color = tuple(color)
                    ax.bar(i, 1, color=rgb_color)
                ax.set_xticks(range(n_segments))
                ax.set_xticklabels([f'Segment {i+1}' for i in range(n_segments)])
                ax.set_yticks([])
                st.pyplot(fig)
                
                # Download segmented image
                segmented_pil = Image.fromarray(segmented)
                img_byte_arr = io_lib.BytesIO()
                segmented_pil.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                st.download_button(
                    label="Download Segmented Image",
                    data=img_byte_arr,
                    file_name="segmented_image.png",
                    mime="image/png"
                )
        else:
            st.warning("Please upload a color image.")

if __name__ == "__main__":
    run() 