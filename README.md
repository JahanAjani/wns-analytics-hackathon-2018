# Employee_promotion_Prediction_using_CatBoost
Employee_promotion_Prediction_using_CatBoost

<h1>1. Business Problem</h1>

<h2>1.1. Description</h2>
<p>Your client is a large MNC and they have 9 broad verticals across the organisation. One of the problem your client is facing is around identifying the right people for promotion (only for manager position and below) and prepare them in time. </p>
<p>Currently the process, they are following is:
They first identify a set of employees based on recommendations/ past performance
Selected employees go through the separate training and evaluation program for each vertical. These programs are based on the required skill of each vertical
<p>
At the end of the program, based on various factors such as training performance, KPI completion (only employees with KPIs completed greater than 60% are considered) etc., employee gets promotion
For above mentioned process, the final promotions are only announced after the evaluation and this leads to delay in transition to their new roles. Hence, company needs your help in identifying the eligible candidates at a particular checkpoint so that they can expedite the entire promotion cycle.
<p>
They have provided multiple attributes around Employee's past and current performance along with demographics. Now, The task is to predict whether a potential promotee at checkpoint in the test set will be promoted or not after the evaluation process.
<p>
  
<h2>1.2. Source/Useful Links</h2>
<p>https://datahack.analyticsvidhya.com/contest/wns-analytics-hackathon-2018/</p>
<h1>2. Machine Learning Problem Formulation</h1>
<h2>2.1. Data</h2>

<h3>2.1.1. Data Overview</h3>

- Source: https://datahack.analyticsvidhya.com/contest/wns-analytics-hackathon-2018/
<p>WNS Analytics Wizard 2018 Data hack competition from analyticsvidhya.com, same data set i am using</p>
<p>Training Data: it have 54808 records and 14 columns</p>
<p>Test Data: it have 23490 records and 13 columns</p>
 
<h2>2.2. Mapping the real-world problem to an DL problem</h2>
<h3>2.2.1. Type of Deep Learning Problem</h3>

<p>Binary Classification :</p>
- Based on Employee's past and current performance along with demographics. Now, The task is to predict whether a potential promotee at checkpoint in the test set will be promoted or not after the evaluation process.:

<h3>2.2.2. Performance Metric</h3>

Metric(s): F1 score 
<p>
 data set have unbalanced data[9:1] so better to select F1 score instead of Accuracy
</p>
<h2>
2.3. Train, CV and Test Datasets
</h2>
Split the Training  dataset  into Two parts train, and cross validation with 70% and 30% of data respectively

<h2>3. Code</h2>
<p>Main code file is <h4>wns_predicitng_potential_employee.ipynb </h4></p>
