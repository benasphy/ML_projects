# Hierarchical Clustering Projects

This folder contains various projects that utilize Hierarchical Clustering for different applications. Each project is designed to demonstrate the use of hierarchical clustering in machine learning tasks with interactive visualizations.

## Projects

1. **Document Clustering**: Clusters text documents based on their content using hierarchical clustering. Features include:
   - Interactive parameter tuning
   - Dendrogram visualization
   - Document similarity matrix
   - Word cloud visualization
   - Cluster analysis and interpretation

2. **Market Basket Analysis**: Analyzes shopping patterns using hierarchical clustering. Features include:
   - Transaction clustering
   - Item correlation analysis
   - Association rules mining
   - Interactive visualizations
   - Cluster analysis and interpretation

## How to Run

To run any of the projects, follow these steps:

1. Ensure you have the required dependencies installed. You can install them using pip:

   ```bash
   pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn wordcloud mlxtend
   ```

2. Navigate to the Hierarchical directory in your terminal.

3. Run the Streamlit app using the following command:

   ```bash
   streamlit run main.py
   ```

4. Use the sidebar to select the project you want to run.

## Project Structure

- `main.py`: The main entry point for running the projects.
- `Hierarchical_projects/`: Contains individual project files:
  - `document_clustering.py`: Document clustering project.
  - `market_basket_analysis.py`: Market basket analysis project.

## Features

- Interactive parameter tuning
- Real-time visualizations
- Detailed cluster analysis
- Support for custom data input
- Rich visualization tools including:
  - Dendrograms
  - Heatmaps
  - Scatter plots
  - Word clouds
  - Association rule visualizations

## Contributing

Feel free to contribute to these projects by submitting pull requests or opening issues for any bugs or feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 