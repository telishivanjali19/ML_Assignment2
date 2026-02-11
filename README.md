# ML_Assignment2
Mobile Price Classification using Machine Learning

⸻

a) Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict the price category of a mobile phone based on its technical specifications.

Given various features such as battery power, RAM, screen resolution, internal memory, etc., the model must classify the mobile phone into one of four price categories:
	•	0 → Low Cost
	•	1 → Medium Cost
	•	2 → High Cost
	•	3 → Very High Cost

The project demonstrates end-to-end machine learning workflow including model building, evaluation, comparison, and deployment using Streamlit.

⸻

b) Dataset Description  (1 Mark)

The dataset used is the Mobile Price Classification Dataset obtained from Kaggle.
	•	Total Instances: 2000
	•	Total Features: 20 numerical features
	•	Target Variable: price_range (4 classes)

Features Include:
	•	battery_power
	•	blue
	•	clock_speed
	•	dual_sim
	•	fc (front camera)
	•	four_g
	•	int_memory
	•	m_dep
	•	mobile_wt
	•	n_cores
	•	pc (primary camera)
	•	px_height
	•	px_width
	•	ram
	•	sc_h
	•	sc_w
	•	talk_time
	•	three_g
	•	touch_screen
	•	wifi

All features are numerical, making preprocessing simpler.

The dataset satisfies assignment requirements:
	•	Minimum 500 instances ✔
	•	Minimum 12 features ✔

⸻

c) Models Used and Evaluation Metrics  (6 Marks)

The following six classification models were implemented on the same dataset:
	1.	Logistic Regression
	2.	Decision Tree Classifier
	3.	K-Nearest Neighbors (KNN)
	4.	Naive Bayes (Gaussian)
	5.	Random Forest (Ensemble Model)
	6.	XGBoost (Ensemble Model)

Evaluation Metrics Used:

For each model, the following performance metrics were calculated:
	•	Accuracy
	•	AUC Score (One-vs-Rest for Multiclass)
	•	Precision (Weighted)
	•	Recall (Weighted)
	•	F1 Score (Weighted)
	•	Matthews Correlation Coefficient (MCC)


📊 Comparison Table of All Models

ML Model Name	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.9750	0.999591	0.975946	0.9750	0.975020	0.966875
Decision Tree	0.8375	0.889755	0.843090	0.8375	0.837224	0.784773
KNN	0.5300	0.762890	0.569762	0.5300	0.540707	0.378880
Naive Bayes	0.7975	0.955970	0.806132	0.7975	0.799422	0.731329
Random Forest	0.9025	0.984430	0.904446	0.9025	0.902951	0.870147
XGBoost	0.9050	0.991322	0.906155	0.9050	0.905007	0.873522

Observations

ML Model Name	Observation about Model Performance
Logistic Regression	Achieved the highest accuracy (97.5%) and MCC (0.9668), indicating excellent class separation and strong linear decision boundaries in the dataset.
Decision Tree	Moderate performance with 83.75% accuracy; slightly lower AUC suggests limited generalization compared to ensemble methods.
KNN	Performed poorly (53% accuracy), possibly due to sensitivity to high-dimensional feature space and choice of k value.
Naive Bayes	Achieved reasonable AUC (0.9559) but lower accuracy due to strong independence assumption among features.
Random Forest	Strong performance (90.25% accuracy) due to ensemble averaging reducing overfitting compared to Decision Tree.
XGBoost	High AUC (0.9913) and strong overall performance; slightly below Logistic Regression in accuracy but very stable.

Deployment Details

The models were deployed using Streamlit Community Cloud.

The web application includes:
	•	CSV dataset upload option
	•	Model selection dropdown
	•	Display of evaluation metrics
	•	Confusion matrix output
