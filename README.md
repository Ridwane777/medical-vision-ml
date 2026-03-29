# Project Setup

This document explains how to install the required tools and dependencies to run the project using **Python and Jupyter Notebook**.

---

# 1. Install Python

Make sure Python is installed on your system.

Check your Python version:

```bash
python3 --version
```

You should have **Python 3.9 or newer**.

If Python is not installed, download it from:

https://www.python.org/downloads/

---

# 2. Create a Virtual Environment (Recommended)

Creating a virtual environment prevents dependency conflicts.

Create the environment:

```bash
python3 -m venv venv
```

Activate it
```bash
source venv/bin/activate

---

Install the required libraries using pip:

```bash
pip install -r requirements.txt
```


# 4. Launch Jupyter Notebook

Run the following command:

```bash
jupyter notebook
```

Your browser will open automatically.

You can now open the project notebooks.

---

# 5. Project Notebooks

Run the notebooks in the following order:

```
supervised_learning/dataset_knee_osteoarthritis.ipynb

unsupervised_learning/Student_Dataset_preprocessing.ipynb
```

---

# 6. Deactivate the Environment

When you are done working:

```
deactivate
```

---

# Required Libraries

The project uses the following tools:

* **NumPy** – numerical computations
* **Pandas** – data manipulation
* **Matplotlib** – data visualization
* **Scikit-learn** – machine learning models
* **Jupyter Notebook** – interactive development environment
