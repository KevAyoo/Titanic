# Titanic
A machine learning project using Logistic Regression to predict Titanic survival outcomes with 84% accuracy based on passenger demographics and family structures.

Predicting Survival with 84% Accuracy
This project explores the famous Titanic dataset to predict passenger survival. Using a combination of Exploratory Data Analysis (EDA), Feature Engineering, and Supervised Machine Learning, I developed a model capable of predicting whether a passenger survived the sinking of the Titanic based on characteristics like age, class, and family size.

The Data Science Workflow

 Exploratory Data Analysis (EDA)
I began by visualizing the hidden patterns in the data:
•	Gender: Confirmed that females had a significantly higher survival rate.
•	Social Class: Observed a clear survival advantage for 1st Class passengers.
•	Family Size: Discovered that passengers traveling in small families (2-4 people) survived at higher rates than those alone or in very large families.
•	Age: Identified a "survival bump" for young children, specifically those with the title "Master".

2. Feature Engineering
To improve model accuracy, I transformed raw data into high-signal features:
•	Title Extraction: Pulled titles (Mr, Mrs, Miss, Master) from names to capture social status and age groups.
•	Cabin Mapping: Created a binary HasCabin feature, as passengers with recorded cabin numbers had a 0.31 correlation with survival.
•	Family Grouping: Categorized passengers into "Alone," "Small," and "Large" groups to capture non-linear survival trends.
•	Fare Binning: Divided the skewed fare data into quartiles (Low to Very High).

Model Selection & Evaluation
I tested four different algorithms to find the most effective "brain" for this data:
1.	Logistic Regression (The Winner)
2.	Support Vector Machine (SVM)
3.	Random Forest

Technologies Used
•	Python (Pandas, NumPy)
•	Matplotlib & Seaborn (Data Visualization)
•	Scikit-Learn (Machine Learning & Scaling)

Conclusion
This project demonstrates that even with a simple model like Logistic Regression, we can achieve high accuracy by performing deep feature engineering. The most significant barrier to survival was not just "luck," but rather social class and the logistical difficulty of navigating a large family during the evacuation.

