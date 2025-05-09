# Machine Learning Projects Collection

A comprehensive collection of machine learning projects implemented in Python, covering various algorithms and techniques. Each project is designed to solve real-world problems using different machine learning approaches.

## Project Categories

### Supervised Learning
- **Linear Regression**
  - House Price Prediction
  - Salary Prediction
  - Study Hours vs Exam Score Prediction
  - Messi Goal Prediction
  - Normal Equation vs Gradient Descent Implementation

- **Logistic Regression**
  - Diabetes Prediction
  - Rock vs Mine Classification
  - Simple HIV Prediction

- **Naive Bayes**
  - Fake News Detection
  - Spam Detection
  - Weather Prediction

- **Support Vector Machine (SVM)**
  - Breast Cancer Prediction
  - Spam Detection

- **K-Nearest Neighbors (KNN)**
  - Movie Recommendation System
  - T-Shirt Size Prediction

- **Decision Trees**
  - Gym Decision Tree
  - Gini Impurity Implementation

### Unsupervised Learning
- **Clustering**
  - **K-Means**
    - Customer Segmentation
    - Loan Approval Clustering

  - **Gaussian Mixture Models (GMM)**
    - Customer Segmentation
    - Image Color Segmentation

  - **DBSCAN/HDBSCAN**
    - Customer Behavior Analysis
    - Anomaly Detection

  - **Hierarchical Clustering**
    - Document Clustering
    - Market Basket Analysis

  - **Fuzzy C-Means**
    - Customer Profiling
    - Image Segmentation

### Other Techniques
- **Dimensionality Reduction**
  - Feature Selection
  - Image Compression

- **Association Rule Learning**
  - Market Basket Analysis
  - Recommendation System

- **Poisson Regression**
  - Competition Award Prediction
  - Car Accident Prediction

## Project Structure

Each project category has its own directory containing:
- `main.py`: Main entry point for running the Streamlit app
- `requirements.txt`: Required Python packages
- Project-specific files and datasets
- Detailed README.md with project documentation

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/benasphy/ML_projects.git
   cd ML_projects
   ```

2. Install dependencies for a specific project:
   ```bash
   cd <project_directory>
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

## Common Requirements

Most projects require these Python libraries:
- `streamlit`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `plotly`

Additional requirements are specified in each project's `requirements.txt` file.

## Features

- Interactive web interfaces using Streamlit
- Real-time data visualization
- Model evaluation and metrics
- Custom dataset support
- Comprehensive documentation
- Clean and modular code structure

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Datasets used in these projects are sourced from publicly available repositories
- Special thanks to the contributors of the Python libraries used in these projects
- Inspired by various machine learning courses and tutorials

---
Feel free to star the repository if you find it useful! 