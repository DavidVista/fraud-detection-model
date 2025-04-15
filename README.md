# fraud-detection-model
This project aims to detect fraud transactions by combining Nature-Inspired Computing (NIC) algorithms (e.g., Particle Swarm Optimization, Ant Colony Optimization) with traditional Machine Learning models (e.g., Random Forest, XGBoost, Neural Networks).

# Checkpoint I
During the first several weeks, we loaded the dataset and prepared data for further manipulation. Specifically, the dataset was loaded, and the tables were merged for further feature engineering. You can see the <a href="https://github.com/DavidVista/fraud-detection-model/blob/exploration/data_extraction.ipynb">notebook</a> in the branch <a href="https://github.com/DavidVista/fraud-detection-model/tree/exploration">exploration</a> to follow the next steps that have been made:
- Imputing missing values;
- Distribution analysis;
- Removing redundant features;
- Encoding categorical features;
- Creating a learning baseline for the model.

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
    <li>RNN: learning_rate, RNN units, dropout rate.</li>
  </ul>
</ul>
