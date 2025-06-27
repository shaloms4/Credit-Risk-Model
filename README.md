# Credit-Risk-Model

## Credit Scoring Business Understanding

## Exploratory Data Analysis (EDA) Insights

1. **Data Completeness and Quality**: The dataset contains no missing values in any column, ensuring a clean foundation for modeling and eliminating the need for imputation or special handling of missing data.

2. **Feature Distributions and Outliers**: Numerical features such as `Amount` and `Value` exhibit skewed distributions and contain outliers, as revealed by histograms and boxplots. Categorical features show varying frequencies, with some dominant and some rare categories, which may impact model encoding and performance.

3. **Relationships Between Features**: Correlation analysis among numeric features highlights how variables relate to each other. For example, strong or weak correlations can inform feature selection and engineering for modeling.

### 1. Basel II and the Need for Interpretable, Documented Models

The Basel II Accord emphasizes accurate **risk measurement** to ensure banks hold enough capital against potential losses. This regulatory focus requires credit scoring models to be **interpretable**, **transparent**, and **well-documented** so that decisions (like loan approvals or rejections) can be clearly explained to regulators, auditors, and stakeholders. Black-box models without clear reasoning can lead to compliance issues and undermine trust.

### 2. Importance and Risks of Proxy Variables

In the absence of a direct "default" label (i.e., an exact indicator of loan non-repayment), we must create a **proxy variable**—an approximate definition of default (e.g., "missed 3 payments in 6 months"). This is essential for training and evaluating the model. However, using a proxy introduces **business risks** such as mislabeling borrowers, inaccurate risk estimation, and poor decision-making. These can lead to financial losses or unfair loan rejections.

### 3. Trade-offs: Simple vs. Complex Models

Simple models like **Logistic Regression with Weight of Evidence (WoE)** offer high interpretability and regulatory compliance, making them easier to audit and explain. In contrast, complex models like **Gradient Boosting** may deliver higher predictive performance but often lack transparency, which can raise concerns in regulated environments. The trade-off lies between **accuracy** and **accountability**—banks must choose models that balance performance with the ability to justify decisions.
