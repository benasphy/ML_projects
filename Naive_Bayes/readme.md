# Naive Bayes Projects

This folder contains various projects that utilize the Naive Bayes algorithm for different applications. Each project is designed to demonstrate the use of Naive Bayes in machine learning tasks.

## Projects

1. **Weather Prediction**: Predicts weather conditions using historical data.
2. **Spam Detection**: Classifies emails as spam or not spam using text data.
3. **Fake News Prediction**: Detects fake news articles using features like title, news URL, source domain, and tweet number.

## How to Run

To run any of the projects, follow these steps:

1. Ensure you have the required dependencies installed. You can install them using pip:

   ```bash
   pip install streamlit pandas scikit-learn
   ```

2. Navigate to the Naive_Bayes directory in your terminal.

3. Run the Streamlit app using the following command:

   ```bash
   streamlit run main.py
   ```

4. Use the sidebar to select the project you want to run.

## Project Structure

- `main.py`: The main entry point for running the projects.
- `Naive_Bayes_projects/`: Contains individual project files:
  - `weather_prediction.py`: Weather prediction project.
  - `spam_detection_nb.py`: Spam detection project.
  - `fake_news_prediction.py`: Fake news prediction project.

## Data

Each project uses its own dataset, which is either uploaded by the user or loaded from a default CSV file located in the `Naive_Bayes_projects/` directory.

## Contributing

Feel free to contribute to these projects by submitting pull requests or opening issues for any bugs or feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
