# Transforming Business Operations Through AI: A Framework for Effective Integration

**Author:** Alex Brogan  
**Institution:** Hiram College  
**Degree:** B.S. Applied Computer Science  
**Year:** 2025

---

## Overview

This senior capstone project investigates the integration of artificial intelligence into modern business operations, with a focus on predictive modeling, data analytics, and strategic decision-making. The project combines an extensive literature review with a hands-on machine learning implementation to demonstrate how AI can forecast consumer purchasing behavior and inform business strategy across departments.

The research identifies key benefits of AI adoption — including productivity gains, cost reduction, and improved customer experiences — while also addressing ethical considerations, implementation barriers, and workforce adaptation challenges.

---

## Research Questions

- How effectively can AI predict consumer purchasing behavior from raw shopping trend data?
- How can businesses leverage predictive analytics to improve operational decisions across departments?
- What framework best supports effective AI integration into existing business practices?

---

## Project Deliverables

- **Research Paper** — Full academic paper with literature review, methodology, results, and conclusions
- **Final Report** — Summary of findings and model outcomes
- **Predictive Model** — Python-based machine learning implementation
- **Presentations** — Progress and final presentations delivered throughout the project lifecycle
- **Abstract** — Condensed overview of the project scope and findings
- **Annotated Bibliography** — Curated research sources supporting the project

---

## Technical Implementation

### Dataset
- **Source:** `shopping_trends.csv` (Kaggle — Srestha Jain)
- **Size:** ~3,900 consumer records
- **Features:** Demographic attributes, browsing history, product category preferences, past purchase frequency, and purchase amounts
- **Split:** 80/20 train/test (3,120 training records, 780 testing records)

### Model Development

**Phase 1 — Regression Approach (Random Forest Regressor)**  
The initial model attempted to predict exact purchase amounts. Results were suboptimal:
- RMSE: $23.92
- R-squared: -0.02

Hyperparameter tuning and additional preprocessing yielded only marginal improvements, indicating the regression approach was not well-suited for this dataset.

**Phase 2 — Classification Approach (Random Forest Classifier)**  
The project pivoted to predicting purchase *likelihood* (binary classification) rather than purchase amount. This reformulation produced significantly better results:
- Accuracy: 100%
- Precision: 1.00
- Recall: 1.00
- F1-Score: 1.00

Results held consistent across multiple data splits (80/20, 70/30, 60/40), with only the support values varying between trials.

### Neural Network (PyTorch)
A feedforward neural network was also implemented using PyTorch for binary classification:
- Architecture: Input layer → 64 nodes → 32 nodes → Output (Sigmoid)
- Optimizer: Adam (lr=0.001)
- Loss function: Binary Cross Entropy (BCELoss)
- Batch size: 16, 50 epochs

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML Libraries | PyTorch, TensorFlow, Scikit-Learn |
| Data Processing | Pandas, NumPy |
| Visualization | Microsoft Visual Basic |
| Dataset | shopping_trends.csv (Kaggle) |

---

## Key Findings

- A classification-based approach significantly outperformed regression for predicting consumer purchasing behavior on this dataset
- AI-driven predictive analytics can directly inform business decisions across marketing, supply chain, manufacturing, R&D, and customer service
- The model demonstrated that consumer attributes such as age, product category preferences, and past purchase frequency are strong predictors of purchase likelihood
- Perfect classification results (F1 = 1.00) were achieved, though future work should validate against additional datasets to assess generalizability and rule out overfitting

---

## Business Implications

The project analyzed how predictive AI data cascades across business departments:

- **Marketing** — Targeted advertising, personalized promotions, and pricing strategy informed by purchase likelihood data
- **Management** — Staffing forecasting, AI chatbot deployment for customer service cost reduction
- **Supply Chain** — Demand forecasting, lead time planning, and inventory optimization
- **Manufacturing** — Automation of repetitive tasks, reduced error rates, and workforce cost efficiency
- **R&D** — Product development informed by purchase trends and customer reviews

---

## Future Work

- Expand the dataset with additional attributes such as real-time browsing behavior and post-purchase reviews
- Deploy the classification model as a REST API for integration with existing business intelligence platforms
- Explore deep learning and reinforcement learning techniques for adaptive, real-time decision-making
- Validate the model against external datasets to confirm generalizability

---

## References

- Jain, S. Shopping Trends Dataset. Kaggle.
- Manyika, J., et al. (2018). Notes from the AI frontier. McKinsey Global Institute.
- PwC. (2020). AI predictions: What's next for AI and business?
- Davenport, T. H., & Mittal, N. (2021). AI-driven innovation. Harvard Business Review.
- Huang, M. & Rust, R. T. (2022). Artificial Intelligence in Service. Journal of Service Research.
- Grewal, D., et al. (2021). The Impact of AI on Consumer Behavior. Journal of Retailing.
- Adam, M., et al. (2022). The Role of AI Chatbots in Customer Service. Computers in Human Behavior.
- Zanker, M. & Jannach, D. (2020). Recommender Systems. Springer.
- Zhou, X. & Pan, J. (2021). AI Sentiment Analysis in Business Decision Making. Information Systems Journal.

---

## Author

**Alex Brogan**  
B.S. Applied Computer Science | Hiram College, 2025  
Minors: Entrepreneurship & Music Composition  
[alex.brogan10@gmail.com](mailto:alex.brogan10@gmail.com) | [github.com/alexbrogan](https://github.com/alexbrogan)
