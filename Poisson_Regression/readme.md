# Poisson Regression Projects

This repository contains various Poisson Regression projects implemented in Python. Each project demonstrates the application of Poisson Regression to solve real-world problems using datasets.

## Project Structure

```
Poisson_Regression/
├── main.py
├── requirements.txt
├── Poisson_regression_projects/
│   ├── competition_award.py
│   ├── no_of_car_accident.py
│   ├── competition_awards_data.csv
```

### Key Files
- **`main.py`**: The main entry point for running the Streamlit app.
- **`requirements.txt`**: Contains the dependencies required to run the project.
- **`Poisson_regression_projects/`**: Contains individual project scripts and datasets.

## Projects Included

1. **Competition Award Prediction**  
   Predicts the number of awards a student will receive based on their math scores using Poisson Regression.  
   Dataset: `competition_awards_data.csv`

2. **Number of Car Accidents Prediction**  
   Predicts the number of car accidents based on average speed, traffic density, and road conditions using Poisson Regression.  
   Dataset: Synthetic data (hardcoded in the script).

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/benasphy/ML_projects.git
   cd Poisson_Regression
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

## Datasets

- **`competition_awards_data.csv`**: Contains data for predicting the number of awards based on math scores.

## Screenshots

Add screenshots of the Streamlit app interface here to showcase the projects.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Datasets used in this project are sourced from publicly available repositories.
- Special thanks to the contributors of the Python libraries used in this project.

---
Feel free to contribute to this repository by submitting issues or pull requests.
