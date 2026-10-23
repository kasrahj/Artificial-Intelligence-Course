# Artificial Intelligence Course Projects

Six notebook assignments exploring search, reinforcement learning, probabilistic models, supervised learning, deep learning, and unsupervised learning. The emphasis is on implementing algorithms, comparing approaches, and analyzing their behavior on concrete datasets and environments.

## Technical work

| Assignment | Implementation and methods |
| --- | --- |
| [CA1: Genetic Algorithms](<CA1-Genetic Algorithms/curve fitting problem.ipynb>) | Encodes candidate polynomial coefficients as chromosomes and evolves a population against five target points. The notebook implements initialization, an absolute-error fitness measure, selection, crossover, mutation, and elitist carry-over. The configured run uses a population of 200 for up to 1,000 generations, with crossover, mutation, and carry probabilities set in the notebook. |
| [CA2: Reinforcement Learning](<CA2-Reinforcement Learning/AI_CA2.ipynb>) | Implements value iteration and policy iteration for the 4×4 FrozenLake environment, then builds a tabular Q-learning agent for Taxi-v3. Experiments compare learning behavior under fixed and decaying learning rates and use seeded runs for repeatability. |
| [CA3: Hidden Markov Models](<CA3-Hidden Markov Models/AI_CA3.ipynb>) | Classifies blues, hip-hop, metal, and pop from audio sequences. It derives MFCC, delta, and delta-delta features, trains four-state HMMs with `hmmlearn`, and compares them with a separate HMM implementation. Songs are split into fixed-length segments; the notebook reports classification results and confusion-matrix analysis. |
| [CA4: Machine Learning](<CA4-Machine Learning/AI-CA.ipynb>) | Uses a marketing-campaign dataset to work through missing-data handling, correlation and feature analysis, normalization, train/test validation, and k-fold cross-validation. It implements linear and multiple regression with gradient descent, then explores decision trees, random forests, hyperparameter search, and the effect of Laplace-noise perturbation on model performance. |
| [CA5: Convolutional Neural Networks](<CA5-Convolutional Neural Networks/AI_CA5.ipynb>) | Builds a PyTorch image pipeline for grayscale 512×512 MRI images across glioma, meningioma, pituitary-tumor, and no-tumor classes. The notebook defines custom datasets and CNN models, trains with cross-entropy and Adam, and explores regularization techniques such as dropout and batch normalization. |
| [CA6: Clustering](<CA6-Clustering/AI_CA6.ipynb>) | Processes Persian text with Hazm and stop-word filtering, encodes labels, and learns document embeddings with Doc2Vec. It compares K-Means and DBSCAN, evaluates clusters with silhouette and homogeneity measures, and visualizes embeddings after PCA reduction to two and three dimensions. |

## How to explore the work

Open an assignment notebook in Jupyter Notebook or JupyterLab and follow its sections in order. Together, the notebooks show algorithm implementation, parameter selection, evaluation, and visualization—not just calls to a model API. CA2 includes interactive Gym render modes, while CA3, CA4, CA5, and CA6 include dataset-specific analysis and plots.

## Environment and data

The notebooks use Python and packages including NumPy, pandas, Matplotlib, scikit-learn, Gym/Gymnasium, `hmmlearn`, Librosa, PyTorch, Hazm, and Gensim. A basic environment can be started with:

```bash
python -m pip install jupyter numpy pandas matplotlib scikit-learn
```

Install assignment-specific packages as needed. Some data is not included in the repository: CA3 expects genre-organized audio, CA4 expects its marketing dataset, CA5 mounts an MRI dataset from Google Drive, and CA6 reads `train.csv` and `test.csv`. Update notebook paths and install the relevant libraries before running those assignments. Several cells assume Google Colab or particular working directories.

These are course experiments. Their results depend on the input data, random seeds, and package versions; the repository does not bundle every source dataset.
