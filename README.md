# Spam Email Classification - ML Models

A program that can classify emails as spam or not spam using machine learning algorithms.
This project was made during the Compozent internship in Machine Learning and Artificial Intelligence.

![Image 1](./plots/label_dist.png)
*Above chart shows the labels distributions.*

### You need to download these first for NLTK

```Python
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Algorithms Used / Results

* Logistic Regression

```
Classification Report for Logistic Regression: 

              precision    recall  f1-score   support

           0       0.99      0.94      0.97       761
           1       0.95      1.00      0.97       839

    accuracy                           0.97      1600
   macro avg       0.97      0.97      0.97      1600
weighted avg       0.97      0.97      0.97      1600
```
![Image 2](./plots/Logistic_Regression_cm.png)
*Above chart shows the confusion matrix for Logistic Regression*

* Random Forest

```

```

![Image 3](./plots/Logistic_Regression_cm.png)
*Above chart shows the confusion matrix for Random Forest*











