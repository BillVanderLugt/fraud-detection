#### Fraud Detection Case Study

In this project, we were tasked with predicting fraud for an event management
company. Given an extensive data set on past events booked through this company,
we worked to classify these bookings as either fraudulent or not based on the
previously determined account type.

Using a Random Forest model, we train on this provided data in order to predict
on future unseen data.

#### Feature Importance

Looking through the original data, we decided to first focus on only the
numerical columns. The ones of most importance were:

* body_length
* sale_duration2
* user_age
* name_length
* payee_name
* user_type
* fb_published

Knowing the score on our predictions could increase through analyzing the
vocabulary used in the description, we began implementing natural language
processing while also running our numerical columns through a model.

#### Model

The final model used for our fraud predictions is a Random Forest, utilizing
only the 7 numerical features listed above. We hoped to use our natural
language processing to feature engineer a new column of the probability of fraud
based on the vocabulary in the description.

#### Web App

Our model can be run on new data through a convenient wed application hosted on
an EC2 instance on AWS. On this we have included descriptions of the various
steps in our process along with two separate ways to make a prediction via our
model. You may either ping an existing server for a random data point, or
manually input values for the numerical fields of our model. These data points
and predictions will then be stored in a postgres database on the instance.
