# Italian Rap Songs: Data Mining and Predictive Analysis

This project explores the Italian Rap and Trap scene through Data Mining and Machine Learning, focusing on audio features and lyrical content to perform clustering and regional classification.

## Key Contributions

In this project, I was responsible for the following core areas:

* Predictive Analysis (Machine Learning):
    * Designed and implemented the NLP pipeline using a fine-tuned UmBERTo (Italian BERT) model.
    * Implemented a weighted loss function specifically to handle class imbalance between macro-zones.
    * Developed a custom Similarity-based TF-IDF approach for extracting regional linguistic markers using a specialized lexicon.
    * Built the final Stacking Ensemble using XGBoost as a meta-learner, achieving a Macro F1-score of 0.7554.
* Clustering Analysis:
    * Co-produced the clustering pipeline and led the fine-tuning process to optimize cluster stability and separation.
    * Conducted the complete qualitative interpretation of the results, mapping the clusters to specific artistic movements (e.g., Old School vs. Modern Evolution).
    * Validated the clusters through multidimensional analysis, ensuring the consistency of the identified stylistic profiles.
* Data Preparation and Feature Engineering:
    * Co-developed the data cleaning pipeline and engineered domain-specific features such as swear density and words per second.

---
For a detailed mathematical analysis and ethical considerations, please refer to the Full Report (DM_Report_21.pdf).
