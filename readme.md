### Fraud Detection Case Study

In this project, we were tasked with predicting fraud for an event management
company. Given over 14,000 instances of past events booked through the company,
we worked to classify these bookings as either fraudulent or not based on the
previously determined account type.

Using a Random Forest model, we train on the provided data in order to predict
on future unseen data. The model provides promising results and is made accessible via a web app. Our deployed model achieves an f1 score of 0.89. We provide a scalable framework to incorporate natural language processing (NLP) in future models.

The data is proprietary and as such, some details are excluded.

<img alt="Website Screenshot" src="images/website.png" width='400'>

<sub><b>Figure 1: </b> Website homepage </sub> 

#### Feature Importance

For a minimum viable product, we decided to first focus on only the
numerical columns. The ones of most importance were:

* **body_length** (length of the event description)
* **sale_duration2** (days posted)
* **user_age** (days between user sign-up and event post)
* **name_length** (length of event host's name)
* **payee_ind** (computed from the payee_name field; 0 if no payee_name provided)
* **user_type** (integers between 0 and 3; meaning unknown)
* **fb_published** (Facebook published; 0 or 1)

#### Model

The deployed model (`pure_rf_model.pkl`) used for our fraud predictions is a Random Forest, utilizing
only the 7 numerical features listed above. Cross-validation yielded an f1 score of 0.89 and an accuracy of 0.91.

We figured our predictions could increase through analyzing the text data (namely the event description), so we began implementing NLP (TDIDF feeding an SVC) to formulate an initial fraud probability prediction to feed as a feature to our Random Forest. This increased our cross-validated f1 score to 0.92 and accuracy to 0.94 (promising results!) However, we ran into trouble when predicting on new data with event descriptions containing words not in our trained vocabulary. The model incorporating NLP is `model.pkl`, but is not yet ready for deployment.

#### Web App

Our model can be used to predict on new data through a convenient Flask web application hosted on
an EC2 instance on AWS. The website provides an overview of the problem and our process along with two separate ways to make a prediction via our
model.

1. Ping an existing server for a random data point.
2. Manually input values for the numerical fields of our model.

These data points and predictions will then be stored in a PostgresSQL database on the instance.

#### The Team

Our product was deployed by a team of four in under 16 hours. Some readability and tuning was sacrificed in getting our app deployed and proving our MVP.

Meet our team of talented data scientists brought together by the Denver [Galvanize Data Science Immersive](https://www.galvanize.com/denver-golden-triangle/data-science#outcomes):

* [Bill Vander Lugt](https://www.linkedin.com/in/bill-vander-lugt/)
* [Bob Corboy](https://www.linkedin.com/in/roberttcorboy/)
* [Ky Kiefer](https://www.linkedin.com/in/kykiefer/)
* [Steve Harshman](https://www.linkedin.com/in/starshman/)
