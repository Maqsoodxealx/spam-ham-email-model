# Spam/Ham Email Classification using Machine Learning and NLP
A complete end-to-end Machine Learning project that classifies SMS/Email messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) and Scikit-learn.
## Project Overview
Spam emails and SMS messages are one of the most common cybersecurity and communication problems. Automatically detecting unwanted messages helps users avoid scams, phishing attempts, and unnecessary advertisements.
In this project, a complete Machine Learning pipeline was developed to classify messages as Spam or Ham. The model was trained in Google Colab using Python and deployed as an interactive Streamlit application developed in PyCharm.
The project demonstrates the complete data science workflow, including:
•	Data Cleaning
•	Exploratory Data Analysis (EDA)
•	Natural Language Processing (NLP)
•	Feature Engineering
•	Model Training
•	Model Comparison
•	Model Evaluation
•	Model Selection
•	Model Serialization
•	Streamlit Deployment
Project Objectives
•	Build a highly accurate spam detection model.
•	Apply NLP techniques for text preprocessing.
•	Compare multiple Machine Learning algorithms.
•	Select the best-performing model based on evaluation metrics.
•	Deploy the trained model into a user-friendly application.
•	Demonstrate a complete end-to-end Machine Learning workflow.
Repository Structure
•	2nd_SpamHamEmailsModel.ipynb : Complete Machine Learning notebook developed in Google Colab
•	2nd_SpamHamEmailsModel.py : Python script version of the notebook 
•	2nd-SpamHamEmailsTrainedModel.joblib : Saved Multinomial Naive Bayes model
•	vectorizerTfidf.joblib : Saved TF-IDF vectorizer                                      
•	AppSpamHamInPycharm.py : Streamlit application for prediction                         
•	spam.csv : SMS Spam Collection Dataset                                  
•	README.md : Project documentation      


                                  
Dataset
Dataset Name : SMS Spam Collection Dataset
Source : https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
The dataset contains thousands of labeled SMS messages categorized into two classes:
Technologies Used
•	Python	
•	Google Colab
•	PyCharm
•	Streamlit
•	Scikit-learn
•	Pandas
•	NumPy
•	Matplotlib
•	Seaborn
•	NLTK
•	WordCloud
•	Joblib
•	XGBoost
Machine Learning Workflow
1-	Data Collection : The SMS Spam Collection dataset was imported into Google Colab using Pandas.
2-	Data Cleaning : The dataset was cleaned to improve data quality before model training.
The following preprocessing steps were performed:
•	Checked column data types
•	Removed unnecessary columns
•	Renamed columns
•	Encoded labels using LabelEncoder
•	Removed duplicate records
•	Verified dataset dimensions
Final columns : Target  - Spam (1) / Ham (0)    &    Text   (SMS or Email Message)





 3. Exploratory Data Analysis (EDA)
•	Several visualizations were created to better understand the dataset.
•	Class Distribution
•	Count Plot
•	Pie Chart
•	Observations
o	Ham Messages = 87.37%
o	Spam Messages = 12.63%
o	The dataset is imbalanced, with Ham messages significantly outnumbering Spam messages.
Text Feature Engineering : Three numerical features were extracted from every message:
•	Number of Characters
•	Number of Words
•	Number of Sentences
Comparative Analysis : The following visualizations were created:
•	Bar Charts
•	Histograms
•	Pair Plot
•	Correlation Heatmap
Key Findings
•	Spam messages are generally longer than Ham messages.
•	Spam messages contain more words and sentences.
•	Number of characters and words have a very strong positive correlation (0.97).
•	Message length alone cannot perfectly classify Spam messages.
4. Natural Language Processing (NLP)
A custom text preprocessing pipeline was implemented.
Each message undergoes the following transformations:
•	Convert to lowercase
•	Tokenization
•	Remove special characters
•	Remove punctuation
•	Remove English stopwords
•	Apply Porter Stemming
Example  : 
Congratulations!!
You have WON a FREE ticket!!
↓
congratul won free ticket


5. Word Cloud Analysis
 Separate word clouds were generated for both classes.
Frequently Appearing Spam Words : Free, win, claim, call, txt, reply
Frequently Appearing Ham Words : ok, now, go, call, will
Additionally :  Top 20 Spam words, Top 30 Ham words were visualized using bar charts.
Feature Extraction
Machine Learning algorithms cannot process raw text directly.Therefore, messages were converted into numerical vectors using TF-IDF (Term Frequency–Inverse Document Frequency) python TfidfVectorizer(max_features=3000)
The vectorizer retained the 3000 most informative words.
Machine Learning Algorithms Evaluated The Following Algorithms Were Trained And Evaluated:
•	Logistic Regression
•	Linear Support Vector Classifier (LinearSVC)
•	Support Vector Classifier (SVC)
•	Decision Tree
•	K-Nearest Neighbors
•	Random Forest
•	Gradient Boosting
•	AdaBoost
•	Bagging Classifier
•	Extra Trees Classifier
•	Gaussian Naive Bayes
•	Bernoulli Naive Bayes
•	Multinomial Naive Bayes
•	XGBoost
Train-Test Split
The dataset was divided into: 80% Training Data , 20% Testing Data Using python train_test_split()
Model Performance Table Generated By Dictionary, For Loop, Function For Train Different Algorithms, DataFrame
No.	Algorithm	Accuracy Score	Precision
1	MultinomialNB	0.970986	1.000000
2	KNeighborsClassifier	0.905222	1.000000
3	RandomForestClassifier	0.973888	0.982609
4	SVC	0.975822	0.974790
5	ExtraTreesClassifier	0.974855	0.974576
6	LinearSVC	0.977756	0.967480
7	LogisticRegression	0.955513	0.960000
8	XGBClassifier	0.970019	0.957265
9	GradientBoostingClassifier	0.950677	0.930693
10	BaggingClassifier	0.958414	0.868217


Selected Model  :   Multinomial Naive Bayes + TF-IDF
Although several algorithms achieved slightly higher accuracy, Multinomial Naive Bayes was selected because it provided the highest precision (100%) while maintaining excellent overall accuracy.
Final Model
Algorithm: Multinomial Naive Bayes
Feature Extraction: TF-IDF
Accuracy: 97.10%
Precision: 100%

Model Saving
The trained model and TF-IDF vectorizer were saved using Joblib.
joblib.dump(mnb,"2nd-SpamHamEmailsTrainedModel.joblib")
joblib.dump(tfidf,"vectorizerTfidf.joblib")
Streamlit Application
A user-friendly Streamlit application was developed to classify messages in real time by using Pycharm.
Application Workflow
1. User enters a message.
2. Text is preprocessed.
3. TF-IDF converts text into numerical features.
4. The trained model predicts the class.
5. Prediction is displayed as: This Email is Spam or This Email is not Spam
Installation
Clone the repository : git clone https://github.com/Maqsoodxealx/SpamHamEmailClassification.git
Install dependencies : pip install pandas, numpy, matplotlib, seaborn, nltk, scikit-learn, wordcloud, xgboost, streamlit, joblib
Download NLTK resources : import nltk, nltk.download("punkt") , nltk.download("stopwords")
Run the application : streamlit run AppSpamHamInPycharm.py





Future Improvements
•	Potential enhancements include:
•	Hyperparameter Tuning
•	Cross Validation
•	Deep Learning (LSTM)
•	GRU Networks
•	Transformer Models (BERT, RoBERTa)
•	Flask/FastAPI API Development
•	Docker Containerization
•	Cloud Deployment (Azure, Render, Streamlit Cloud)
Learning Outcomes
This project strengthened my practical understanding of:
•	Data Cleaning
•	Exploratory Data Analysis
•	Natural Language Processing
•	Text Vectorization
•	Feature Engineering
•	Machine Learning Algorithms
•	Model Evaluation
•	Model Selection
•	Streamlit Deployment
•	GitHub Documentation
Author
•	Maqsood Ahmad
•	Education : M.A. Economics , Bachelor of Mathematics
•	By Profession : Teacher
Areas of Interest
•	Machine Learning
•	Deep Learning
•	Artificial Intelligence
•	Natural Language Processing
•	Python Programming
•	Data Science




Acknowledgements
Special thanks to : Zafar Iqbal & Hisham Sarwar
License
This project is created for educational and learning purposes. Feel free to study, modify, and extend the project with appropriate attribution.
If You Like This Project
If you found this repository useful, please consider giving it a ⭐ on GitHub. Your support is greatly appreciated.

