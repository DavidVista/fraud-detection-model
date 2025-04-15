# fraud-detection-model
This project aims to detect fraud transactions by combining Nature-Inspired Computing (NIC) algorithms (Genetic Algorithms (GA) and Particle Swarm Optimization (PSO)) with traditional Machine Learning models (e.g., Random Forest, XGBoost, Neural Networks). Particularly, GA is used for feature selection as the number of attributes in the dataset is large, and not all features are significant for detection. The second optimization is the hyperparameter tuning using PSO. This step helps to find a suboptimal solution to maximize the performance of each model.

# Checkpoint I
During the first several weeks, we loaded the dataset and prepared data for further manipulation. Specifically, the dataset was loaded, and the tables were merged for further feature engineering. You can see the <a href="https://github.com/DavidVista/fraud-detection-model/blob/main/notebooks/01_EDA.ipynb">01_EDA</a> in the folder <a href="https://github.com/DavidVista/fraud-detection-model/tree/main/notebooks">notebooks</a> to follow the next steps that have been made:
- Imputing missing values;
- Distribution analysis;
- Removing redundant features;
- Encoding categorical features;
- Normalizing features;
- Saving dataframe in a parquet.

# Checkpoint II
Throughout weeks 9-11, we focused on building the <a href="https://github.com/DavidVista/fraud-detection-model/blob/main/detection_pipeline.py">pipeline</a> with the following components:
- Feature selection (Genetic Algorithm);
- Hyperparameters tuning of models (Particle Swarm Optimization);
- Model training.
  
<h2> The stages are implemented as follows </h2>
<h3> Genetic Algorithm Feature Selection (DEAP Framework) </h3>
<ul>
<li>Uses a binary chromosome representation (1=feature selected, 0=not selected);</li>
<li>Fitness function evaluates feature subsets using RandomForest with ROC-AUC;</li>
<li>Includes penalty for large feature sets to prevent overfitting;</li>
<li>Implements tournament selection, two-point crossover, and bit-flip mutation.</li>
</ul>

See the notebook <a href="https://github.com/DavidVista/fraud-detection-model/blob/main/notebooks/02_feature_selection.ipynb">02_feature_selection</a> for the feature selection process steps and implementation.

<h3> Particle Swarm Optimization for Hyperparameter Tuning (PySwarms Framework) </h3>
<ul>
<li>Custom objective functions for each model type;</li>
<li>3-fold cross-validation for robust evaluation;</li>
<li>Different search spaces for each algorithm:</li>
  <ul>
    <li>RandomForest: number of estimators, maximum depth, minimum samples split, maximum features.</li>
    <li>XGBoost: learning rate, maximum depth, minimum child weight, subsample, etc.</li>
    <li>LightGBM: learning rate, number of leaves, minimum data in a leaf, etc.</li>
    <li>Neural Network: learning rate, layer sizes, dropout rate.</li>
  </ul>
</ul>

See the notebook <a href="https://github.com/DavidVista/fraud-detection-model/blob/main/notebooks/03_model_tuning.ipynb">03_model_tuning</a> for the hyperparameter tuning, model training, and evaluation implementations.

# Final Results
During the last weeks, the pipeline was performed to obtain the results and report the outcomes of this project. The folder <a href="https://github.com/DavidVista/fraud-detection-model/tree/main/results">results</a> contains the hyperparameter setting and model evaluation reports as well as all visualizations from the notebooks. See the pdf <a href="https://github.com/DavidVista/fraud-detection-model/blob/main/report.pdf">report</a> of the project to get more information about the outcomes and the procedures.

### Contribution
- Ilya Grigorev, DS-01, responsible for EDA and model pipeline.
- Salavat Faizullin, DS-01, responsible for data preprocessing and model evaluation.
