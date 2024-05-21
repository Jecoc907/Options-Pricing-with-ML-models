# Executive Summary
## Project Overview

Our aim in this project was to develop statistical and machine learning models to replicate the functionality of the traditional Black-Scholes option pricing formula, specifically for valuing European call options. A European call option grants the holder the right (but not the obligation) to buy an asset at a predetermined price and time. Our central focus was on leveraging training data to construct models, where our first task involved predicting the value of options—a regression problem. Meanwhile, we tackled a classification problem by predicting whether options were over or underestimated, represented by the BS variable. After training our models, we will perform predictions using test data based on our best models.

## Dataset(s)

Throughout the project, we were given two datasets: option_train.csv and option_test_nolabel.csv. The training dataset consists of 5,000 separate option pricing data on the S&P 500. In particular, for each option, we have recorded: 
Value (C): Current option value
S: Current asset value
K: Strike price of option
r: Annual interest rate
tau: Time to maturity (in years) 
BS: The Black-Scholes formula was applied to this data (using some σ) to get	C_pred. And If an option has C_pred – C > 0, i.e., the prediction overestimated the option value, we associate that option with (Over); otherwise, we associate that option with (Under).
The test dataset is similar except it has only 500 options and is missing the Value and BS variables, used for the final prediction.

## Methods & Results

Refining our models' predictive accuracy, we implemented a series of optimization steps. These encompassed Exploratory Data Analysis (EDA), feature selection, and model comparisons using cross-validation. During the EDA phase, we utilized techniques such as summary statistics, data visualizations, and correlation analysis utilizing correlation matrices. Notably, our training dataset exhibited no missing or erroneous entries. However, we identified highly correlated variables and mitigated this issue through feature selection methods including stepwise selection and best subset selection, shrinkage techniques via ridge and lasso regression, and machine learning techniques. Subsequently, we explored a range of eight regression models and six classification models. 

Following meticulous evaluation, we chose the XGBoosting model for regression tasks, achieving an outstanding out-of-sample R squared of 99.81%. For classification, the Random Forest model stood out, boasting the lowest mean classification error (MSE) of 0.06. We favored these models due to their exceptional performance and robustness across diverse scenarios.


# Approaches We Tried

## EDA
To begin our analysis, we first conducted an exploratory data analysis. Looking at our training data, we checked for any null values, created histograms, and made boxplots to understand the nature of our variables (Figure 1). We then found outliers by standardizing each predictor value to their respective z-scores. With a threshold of 3 standard deviations, we found 46 outliers in the ‘K’ vector and 29 outliers in the ‘r’ vector as seen in Figure 2. We then repeated this EDA on our test data, and found 4 outliers for ‘K’, 1 outlier for ‘tau’, and 3 outliers for ‘r’ (Figure 3). We decided to keep all rows for our training and testing dataset despite the outliers because we did not want to leave out any important metrics unknowingly.

## Regression
In our regression analysis, we employed predictors S, K, r, and tau, with Value serving as the response variable. Initially, we divided the dataset into 80% training data and 20% validation data during preprocessing. Notably, we refrained from standardizing or normalizing the predictors, except for ridge and lasso methods, as no significant skewness was observed.

Our investigation began with calculating the in-sample and out-of-sample R-squared using the Black-Scholes model, where we utilized the 10-year standard deviation of the S&P 500 index as sigma and treated it as the benchmark. Surprisingly, multiple linear regression performed worse than the Black-Scholes model in terms of out-of-sample R-squared, prompting us to investigate whether the underperformance was due to an excessive inclusion of features. We then began exploring alternative linear regression models like best subset, ridge, and lasso to address this issue. Despite our efforts, the improvement in out-of-sample R-squared was minimal. This led us to conclude that the underperformance wasn't due to overfitting but rather reflected the models' limited predictive ability.

The marginal disparities between in-sample and out-of-sample R-squared values across all four models further supported this conclusion (Figure 4). We attribute the poor performance to the inherent linearity of these models, contrasting with the Black-Scholes model's ability to capture non-linearities inherent in financial market dynamics.

In pursuit of higher R-squared values, we turned to decision tree regression with ensemble methods, such as bagging, boosting, and random forest models. Setting the tree depth to 6 for all models and adjusting other hyperparameters, these models consistently outperformed the Black-Scholes formula. Notably, the boosting model yielded the highest out-of-sample R-squared, indicating its superiority in predictive accuracy. Despite their complexity sacrificing interpretability, these models exhibited no signs of overfitting (Figure 4), affirming their reliability.

Given the superior performance of the boosting model, we have selected it as our prediction model moving forward.

## Classification
For our classification models, we created a new column in our training dataset by converting the ‘BS’ variable into a binary vector, where we classified overestimation by the Black-Scholes model as 1, and underestimation by the Black-Scholes model as 0. This binary vector was used as our dependent variable for our classification model.

Our analysis incorporated all 4 predictor variables again and used functions imported from the sci-kit learn library. We tested a total of 6 different models. We started with decision trees and then the ensemble methods of bagging, random forest, and boosting. The last two methods we tried were logistic regression and the kNN method. For all of our methods, we used 10-fold cross-validation to find the mean classification error for each model.

We used several methods because usually there are trade-offs for each model. For example, decision trees are highly interpretable but have poor predictive accuracy. This makes way for ensemble methods such as bagging, random forests, and boosting. Ensemble methods seek to reduce model variance through various techniques such as bootstrap sampling, random sampling of predictors, and sequential error correction at the cost of interpretability. We also explored logistic regression for its efficiency and ease of use, but it is limited in its performance in many situations because the model assumes a linear relationship between the independent and dependent variables. Finally, we tried kNN because the model is easy to implement and can handle noisy and nonlinear data, but kNN can be computationally intensive. 

In short, we wanted to comprehensively analyze our classification model by trying various methods, which all have pros and cons. We measured performance based on the mean classification error calculated from 10-fold cross-validation. 

## Summary of Final Approaches

### Regression
We achieved an out-of-sample R-squared of 99.73% in the boosting model with the general setting of hyperparameters, where the tree depth is 6 and the number of iterations is 200. We then attempted to tune these hyperparameters to increase the model's precision. Regarding the tree depth, we observed that the model performs best when the depth is set to 3. As for the number of iterations, we set it to 2000 and implemented early stoppage with a tolerance of 10 iterations. This means that if the RMSE of the model didn’t improve for 10 consecutive iterations, it stopped optimizing gradients for the new model. The iteration process terminated after 806 rounds, resulting in a final model with an in-sample R-squared of 99.94% and an out-of-sample R-squared of 99.81% as seen in Figure 4. The proximity of these two numbers indicates no sign of overfitting. Moreover, given the similarity in the predictor distribution between the training and test datasets, we anticipate the model could also yield high R-squared values on the test data. 

### Classification
The mean classification errors for the models were calculated, revealing values of 0.081 for decision tree, 0.073 for bagging, 0.06 for random forest, 0.106 for boosting, 0.108 for logistic regression, and 0.087 for kNN. Our final results can be seen in Figure 5. Among these, the random forest model had the lowest mean classification error at 0.06, earning it the top position. To further refine the random forest model's efficacy, hyperparameter tuning was conducted via GridSearchCV, focusing on parameters such as the number of estimators, maximum tree depth (longest path from root to leaf node), and the minimum samples required before a split. The results can be seen in Figure 6, where the best hyperparameters identified were n_estimators: 50, max_depth: None, and min_samples_split: 2. These parameters further enhanced the model's predictive accuracy for option price classification.

## Conclusion

In summary, our project effectively applied various statistical and machine learning techniques to predict European call option values, traditionally estimated using the Black-Scholes model. The boosting model, with an out-of-sample R square of 99.81%, and the Random Forest model, with the lowest mean classification error of 0.06, were our top performers. These results demonstrate the potential of machine learning in financial modeling, although caution is advised due to the models’ limitations in handling data scenarios not represented in the training set. 
 
However, we do not recommend using the models for real-world investment decisions, as they are solely based on specific samples. For example, when S is 10 and K is 1000, the option value would realistically be 0 because the gap between the current value and strike price suggests that this call option could never be profitable. In this scenario, the Black-Scholes model, based on financial reasoning, correctly predicts the value as 0, whereas our boosting regression model predicts a value of over 200. This discrepancy arises because there are no such extreme samples in the dataset for the model to learn the pattern. Therefore, before utilizing our model in practice, we suggest verifying whether all the variables fall within the distribution of our training dataset.

Accuracy is often more critical than interpretability in financial modeling for options pricing due to its direct impact on decisions and outputs. Machine learning models potentially outperform Black-Scholes models since they can adapt to complex, non-linear market dynamics and incorporate a broader range of data inputs, capturing market behaviors that Black-Scholes cannot. Applying all four predictor variables–current asset value, strike price, annual interest rate, and time to maturity is crucial since it provides unique insights essential for accurate pricing. However, directly applying this model to predict option values for Tesla stocks, known for its volatility, without additional validation could lead to inaccurate predictions, highlighting the need for models to be tested in similar volatile conditions to ensure reliability.

Further work should focus on testing these models with more extensive datasets and under varied market conditions to ensure their robustness and reliability. Continuous refinement and validation against real-world data are essential to leverage these models effectively in financial decision-making processes, emphasizing the importance of integrating machine learning in contemporary analysis.
