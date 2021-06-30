# Business problem:

Companies usually have a greater focus on customer acquisition than customer retention. However, it can cost anywhere between five to twenty five times more to attract a new customer than retain an existing one. Increasing customer retention rates by 5% can increase profits by 25%, according to a [research](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers) done by Bain & Company.  

Churn is a metric that measures the no. of customers who stop doing business with a company. Through this metric, most businesses would try to understand the reason behind churn numbers and tackle those factors with reactive action plans.

But what if you could identify a customer who is likely to churn and take appropriate steps to prevent it from happening? The reasons that lead customers to the cancellation decision can be numerous, ranging from poor service quality to new competitors entering the market. Usually, there is no single reason, but a combination of factors that result to customer churn.

Although the customers have churned, their data is still available. Through machine learning we can sift through this valuable data to discover patterns and understand the combination of different factors which lead to customer churn.

Our goal in this project is to identify behavior among customers who are likely to churn. Subsequent to that we need to train a machine learning model to identify these signals from a customer before they churn. Once deployed, our model will identify customers who might churn and alert us to take necessary steps to prevent their churn.


# Data exploration

## ![Personal factors](https://i.ibb.co/5LqNr02/personal-factors.png)

`Observations:`

* Gender has no influence on churn.
* Younger people are more likely to churn. This maybe due to senior citizens only using basic phone services or the phone service is facilitated by younger family members.
* Single people are more likely to churn.
* People without dependents are more likely to churn. These is in sync with the previous point. Maybe single people use more value added services, which is a very competitive space for telecom service providers.

![Contract](https://i.ibb.co/TkpT2bZ/contract.png)

`Observations:`

* Maximum people who churn are on a monthly contract and mostly bill in a paperless manner. Maybe these are tech savvy people who switch to a different carrier as soon as they find a better deal.

![Payment](https://i.ibb.co/r3509PR/payment.png)

`Observations:`

* Maximum people who pay electronically, churn. This supports our hypothesis that tech savvy people churn more often.

![Charges](https://i.ibb.co/80yXBzg/charges.png)

`Observations:`

* Most of the people who churn have low total charges with the carrier.
* Some people who churn are customers who have high monthly and total charges. Maybe these are corporate customers who churn when they are offered a more competitive package.

![Duration](https://i.ibb.co/V9fyTpv/duration.png)

`Observations:`

* Maximum customers churn during the early period of their subscription.

![img](https://i.ibb.co/y035fv3/img.png)

`Observations:`

* Customers with single phone service and no internet service churn the most. Maybe these are people who are not very well off.
* Among these customers who have churned, most have never contacted the tech support.

These customers are probably young tech savvy thrifty customers who change the subscription as soon as they spot a better offer.

![img2](https://i.ibb.co/wYgTP67/img2.png)

`Observations:`

* Maximum people who churn do not stream movies or TV, i.e. they are not dependent on the subscription for streamed media consumption.

# Data preparation

Data is split into train and test sets. The overrepresented class(0) is undersampled using `RandomUnderSampler`.

# Model training

Five models were trained: 
* Support Vector Machine
* Naive Bayes
* Logistic Regression
* Random Forest
* AdaBoost
* Gradient Boosting Machine

# Model evaluation (sample):

Model|	Revenue saved(Rs.)|	Predicted(True positive)(%)|	Missed(False negative)(%)|	F1 score|	ROC_AUC|
-----|---------------|--------------------------|-----------------------|---------|--------------|
AdaBoost|	271500|	84.96|	15.04|	0.607595|	0.774779|
Random Forest|	250000|	84.37|	15.63|	0.602740|	0.770427|
Gradient Boosting Machine|	228000|	83.19|	16.81|	0.601921|	0.768266|
Logistic regression|	226500|	83.19|	16.81|	0.601279|	0.767798|
Naive Bayes|	206000|	86.43|	13.57|	0.571707|	0.748509|
Support Vector Machine|	204500|	82.01|	17.99|	0.600432|	0.765637|

# Business summary

Lost revenue if we do not prevent churn = Rs.9345000 

>Assumed cost of losing a customer: Rs.5000 

>Assumed cost of effort to prevent churn: Rs.1500 



Percentage of customers predicted by 'AdaBoost Classifier' who were going to churn: 84.96%

Percentage of customers missed by 'AdaBoost Classifier' who were going to churn: 15.04%

Revenue saved by preventing churn with our model as compared to no model = Rs 271500



Total expenditure for preventing churn on random 50.0% of customers: Rs.5283000

Extra cost to prevent churn within random 50.0% of the customers = Rs.585667

Our 'AdaBoost Classifier' model saves us Rs.857167 on an average compared to a random selection of 50% customers


# Deployment

The model was deployed using streamlit.

As the users won't check for churn of individual customers, a template was provided which could be downloaded and reuploaded after the data was filled in the indicated format. Then the web app would classify each customer if they would churn.

Here's the app! : https://share.streamlit.io/coderkol95/data-science-projects/churn_app.py
