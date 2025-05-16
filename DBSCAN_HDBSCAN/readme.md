# DBSCAN/HDBSCAN Projects

This repository contains various DBSCAN and HDBSCAN clustering projects implemented in Python. Each project demonstrates the application of density-based clustering algorithms to solve real-world problems using datasets.

## Project Structure

```
DBSCAN_HDBSCAN/
├── main.py
├── requirements.txt
├── DBSCAN_HDBSCAN_projects/
│   ├── customer_behavior_analysis.py
```

### Key Files
- **`main.py`**: The main entry point for running the Streamlit app.
- **`requirements.txt`**: Contains the dependencies required to run the project.
- **`DBSCAN_HDBSCAN_projects/`**: Contains individual project scripts.

## Projects Included

1. **Customer Behavior Analysis**  
   Analyzes customer behavior patterns using HDBSCAN clustering to identify distinct customer segments.

   **Screenshots:**
   ![Customer Behavior Analysis](screenshots/customer_behavior.png)

   - Interactive parameter tuning
   - Cluster visualization
   - Behavior pattern analysis
   - Noise point identification

2. **Anomaly Detection**  
   Detects anomalies in data using DBSCAN clustering.

   **Screenshots:**
   ![Anomaly Detection](screenshots/anom_det.png)

   - Outlier detection
   - Cluster visualization
   - Interactive parameter tuning

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/benasphy/ML_projects.git
   cd DBSCAN_HDBSCAN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

4. Select a project from the sidebar to explore its functionality.

## Requirements

The project requires the following Python libraries:
- `streamlit`
- `numpy`
- `pandas`
- `scikit-learn`
- `hdbscan`
- `matplotlib`
- `plotly`

## Datasets

- **`customer_behavior.csv`**: Contains customer behavior data for analysis.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Datasets used in this project are sourced from publicly available repositories.
- Special thanks to the contributors of the Python libraries used in this project.

---
Feel free to contribute to this repository by submitting issues or pull requests. 