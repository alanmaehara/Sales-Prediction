# From Statistics 101 to Production: a Sales Prediction Project for Rossmann
<img src="https://insideretail.asia/wp-content/uploads/2020/09/Rossmann.jpg" alt="drawing" width="100%"/>

_This sales prediction project uses data from Rossmann, a Germany-based drug store chain with operations over more than 3,000 stores across seven european countries. The dataset is publicly available from a [Kaggle competition](https://www.kaggle.com/c/rossmann-store-sales/data)._

For starters in the Data Science field, it is not always clear how to build end-to-end solutions to solve complex problems involving data. Understanding how to tackle such challenges is key, so does understanding the reasoning behind each step. With this in mind, this project have four goals:

1. Tackle a classic Data Science problem faced by many firms: sales prediction.
2. Go through the statistics theory behind our project steps. We will go deep enough to make sense of the concepts, but won't reach to a point where we bog ourselves down in the road.
3. Unlock insightful information for the business by doing a thorough, detailed data study over the company's data.  
4. Deploy a complete solution suitable for the business needs. Here, we will go from understanding the business demands and prepare the data, to Machine Learning modeling and posterior production.




#### Special Mention

This project was born from [Meigarom Lopes](https://github.com/Meigarom)'s course _Data Science em Produção_. In this course, learners apply a data science solution from scratch from a business perspective. The course goes through the design and implementation of a machine learning project departing from data collection to deployment - for more details, please check his course (in portuguese) [here](https://sejaumdatascientist.com/como-ser-um-data-scientist/).

#### How to read this README.MD?

This is a very extensive README since it carries the responsibility of showing the project outcomes and also to explain the statistics under the hood of our algorithms and analyses. There is also a somewhat palatable length of explanations on the reasoning of each step in this project. Here is my suggestion for the readers:

1. If you are an **experienced reader and don't mind going directly to a notebook**, access the notebook [here](https://github.com/alanmaehara/Sales-Prediction/blob/master/notebooks/cycle02_rossmann_sales_prediction.ipynb) and go back to the [table of contents](#table-of-contents) whenever some explanation is lacking there. 
2. If you wish to **read the project's main findings instead of going through the entire project**, look no further and [get there.](#)
3. **If you wanna bear the adventure with me**, just read the entire readme. A [(go to next section)]() hyperlink will be available for each section and every technical explanation to make your reading smoother. Codes for this project can be found [here](https://github.com/alanmaehara/Sales-Prediction/blob/master/notebooks/cycle02_rossmann_sales_prediction.ipynb).


I would appreciate any comments or suggestions on how to improve this project. Please feel free to add me on GitHub or [Linkedin](https://www.linkedin.com/in/ammaehara/) - I will get back to you as soon as possible. 

With no further due, let's get started!

---
## Table of Contents
- [Brief Intro - Dirk Rossmann GmbH](#brief-intro-dirk-rossmann-GmbH)
- [Main Findings](#main-findings)
- [Project Methodology](#project-methodology)
- [Cycle Description](#cycle-description)
- [01. A Business Request](#01-a-business-request)
- [02. Data Preparation](#02-data-preparation)
- [03. Feature Engineering](#03-feature-engineering)
- [04. Exploratory Data Analysis (EDA)](#04-exploratory-data-analysis-eda)
- [05. Data Preprocessing](#05-data-preprocessing)
- [06. Feature Selection](#06-feature-selection)
- [07. Machine Learning Modelling](#07-machine-learning-modelling)
- [08. Hyperparameter Tuning](#08-hyperparameter-tuning)
- [09. Error Interpretation & Business Performance](#09-error-interpretation-&-business-performance)
- [10. Deploy Machine Learning Model to Production](#010-deploy-machine-learning-model-to-production)
- [11. A Sales Predictor Bot](#11-a-sales-predictor-bot)
- [Conclusion](#conclusion)
- [Appendix I - Datasets](#appendix-i-datasets)
- [Appendix II - References](#appendix-i-references)

---
## Brief Intro - Dirk Rossmann GmbH

[(go to next section)](#main-findings)

Dirk Rossmann GmbH (Rossmann) is a private, German drug store chain founded in 1972 and is a key player on the European pharmacy market, with operations in healthcare and beauty retail industries. According to [Bloomberg](https://www.bloomberg.com/profile/company/122549Z:GR), Rossmann offers a wide range of products including baby and body care, hygiene, sun protection, cosmetics, dental hygiene, household, pets, hair care, perfume, fragrances, and food products. 

 Aside from the +2,000 on-site Germany stores (see stores location [here](https://www.rossmann.de/de/filialen/index.html)), Rossmann operations extend to Poland, Czech Republic, Turkey, Albania and Hungary, totaling +4,100 on-site stores. 
 
 Rossmann is also active on e-commerce for Germany-based customers, with around [$30 million EUR in online revenues per year](https://www.world-today-news.com/the-giant-of-cosmetics-and-drugstore-rossmann-fixed-in-valencia-the-venue-for-its-deployment-in-spain/), making up for 15.2% of market share in Germany. Complete financial data is not publicly available - although [Dun & Bradstreet](https://www.dnb.com/business-directory/company-profiles.dirk_rossmann_gmbh.7342471a5e75a2072c060665843eeecd.html#financials-anchor) claims that Rossmann's 2018 annual revenue was approximately \$9 billion EUR. 

[back to top](#table-of-contents)


---
## Main Findings

[(go to next section)](#project-methodology)

dasdadasdasdas

[back to top](#table-of-contents)

---

## Project Methodology

[(go to next session)](#cycle-description)

For this project, we will use the CRISP-DM as the main method for project management. CRISP-DM stands for "**CR**oss-**I**ndustry **S**tandard **P**rocess for **D**ata **M**ining",  and is considered as one of the gold standards for project management methods in Data Science. For further details on the methodology, Himanshu Shekhar has a [great introduction article](https://medium.com/voice-tech-podcast/cross-industry-standard-process-for-data-mining-crisp-dm-9edc0c5e3a1) to CRISP-DM, although the usage of this method will be easily understood if one follow along with the project's next steps.

The CRISP-DM is a project management methodology that shows a 360º outlook of data science projects. It is composed of six steps that together forms a complete CRISP-DM cycle as follows:

### The CRISP-DM Cycle
  <p>&nbsp;</p>
<img src="https://miro.medium.com/max/700/1*JYbymHifAk7aQ1pHm_IdMQ.png" alt="drawing" width="80%"/>  
<p>&nbsp;</p>


Each cycle is iterative and future cycles serve as a way to improve the current project. There are many benefits for using CRISP-DM as a project management method. Here I highlight three main reasons in favor of CRISP-DM:
*  Delivers an end-to-end solution;
*  Each cycle should be done in a fast-paced. Why? Think of you trying to perfect each and every single step until you are satisfied with the results. It is likely that you will spend weeks (if not months) on a single step, and therefore, won't deliver true value to business since time is usually a constraint;
* By going through a complete cycle, CRISP-DM users can get a grasp of the whole project and map all possible problems on time.

For the purpose of this project, I adapted the CRISP-DM methodology into five steps (instead of six) and allocated each item of the [table of contents](#table-of-contents) on every CRISP-DM step as follows:


**I. Business Understanding**
  * A Business Request

**II. Data Understanding and Data Preparation**
  * Data Preparation
  * Feature Engineering
  * Exploratory Data Analysis (EDA)
  * Data Preprocessing
  * Feature Selection

**III. Modeling**
  * Machine Learning Modelling
  * Hyperparameter Tuning

 **IV. Evaluation**
  * Error Interpretation & Business Performance

 **V. Deployment**

* Deploy Machine Learning Model to Production
* A Sales Predictor Bot


The purpose and application of each step will be shown through the project, so bear with me till the end!
___
## Cycle Description

[(go to next session)](#1-a-business-request)

In this project, you will see the results of the 2nd CRISP-DM cycle. See the log of each cycle below:

| Cycle      | Description | Notebooks |
| ----------- | ----------- | ----------- | 
| 1º      | In this cycle, the main dataset for this project was retrieved from [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data), and complementary datasets related to Germany's economic indicators were retrieved from [OECD.Stat](https://stats.oecd.org/). Customer related data was not utilized in this cycle since such data wouldn't be available at the prediction time.         | [cycle 01](https://github.com/alanmaehara/Sales-Prediction/blob/master/notebooks/cycle01_rossmann_sales_prediction.ipynb)
| 2º   | A few variables were added to the dataset. Number of customers (which was previously dropped from the model) were added. In order to allow this variable into the project, a complementary project to predict number of customers for each Rossmann store was done and predictions were added test dataset. We also remodeled the whole project to attend data leakage issues, although significant effects on performance were not observed.|[complementary project](https://github.com/alanmaehara/Sales-Prediction/blob/master/notebooks/rossmann_customers_prediction.ipynb) & [cycle 02](https://github.com/alanmaehara/Sales-Prediction/blob/master/notebooks/cycle02_rossmann_sales_prediction.ipynb) |


Exact source links of all datasets will be displayed on [Appendix I - Datasets](#appendix_i_datasets). 

[back to top](#table-of-contents)

---

## 01. A Business Request
We start this project with the most important step. Here we understand why a data-driven project needs to be done in first place. There are three tasks to be done:

- **Business Question**: understand the main issue to be solved/question to be answered. Answer the question: "What is the company's main problem and what information addresses this issue (what is the target variable?)
- **Issue Owner and Motive**: get to know who originated the request and why.
- **Solution Format and Deliverables**: identify the type of data problem to be solved. Check possible methods to solve the problem. Define the solution format (how users will access your solution) and granularity (eg: will it be a six-month sales prediction project or six weeks?).

Since we just have the sales dataset from Rossmann and we don't have professional ties with the company, we will create a hypothetical business situation to guide our project. 

### The Business Situation
Let's pretend that we are data scientists working for Rossmann, and that we have just received a business request from three sales managers. They were requesting the exact same thing: a sales forecast for the next six weeks on their respective regional areas. 

Later on, you find out that the CFO was the one who has made this business request to all sales managers. You reached out the CFO and got to know its initial motive: to figure out the total revenue per store after six weeks in order to finance upcoming investments for each store. Then, you suggest a sales forecast project that has as the main output the 6-week sales forecast to be displayed on a smartphone app.

- **Business Question**: what is the sales forecast for the next six weeks?
- **Issue Owner and Motive**: The CFO needs to plan the finance strategy for each store after two months (~6 weeks)
- **Solution Format and Deliverables**:
    - **Data Problem Type**: Sales forecast
    - **Possible Solution Methods**: Regression, Time Series, Neural Networks 
    - **Deliverables**: 6-week daily sales forecast per store. Stakeholders (sales managers, CFO, CEO..) will be able to get forecasts from a smartphone app.

[back to top](#table-of-contents)


---
## 02. Data Preparation
In this step, we work on acquiring data and get first impressions of our problem. Four tasks to be performed:
- **Data Collection**: will you have enough processing and capacity power to acquire the data? Is your data publicly available or do you need to acquire it from the business? Or rather you figure out that you don't have the means to get the necessary data (in this case, the project might not be feasible). Check these points and proceed.  
- **Data Description**: once you get your hands on data for the first time, analyze the dimension of your data (how many rows vs columns?), the data types (categorical data? numerical data? Discrete or continuous data?),
- **Data Cleaning**
    * Null Values: Check the volume of null values on each column of your dataset, and find a method to impute null values. Usually, a good way to determine the best imputation method is to reflect upon the reasons why you have null values for each feature on your dataset. 
    _Quick note: depending on the volume of missing values in the dataset, you might want to go back to the previous task "Data Collection"._
    * Change data types: In python, some features (columns) might not be in the right format to work on (e.g: variable "date" in integer format instead of "date" format). 
- **Descriptive Statistics**: calculate some statistics from your data. Separate numerical and categorical data to perform this step. The recommended statistics are listed below: 
    * Measures of Central Tendency (mean, median)
    * Measures of Dispersion (Variance, Standard Deviation, Range, First and Third Quartiles, Minimum and Maximum)

### [I. Data Collection](#data-collection)

As mentioned in [Cycle Description](#cycle-description), the data comes from a Kaggle competition held by Rossmann. Therefore, our project is pretty limited on the information contained in the dataset. In real life, we would collect all information available in the company's data warehouse that helps answering our [Business Question](#business-question). As for this project, it is fine to proceed as it is, since we are running this project under a fictitional business standpoint.


### [II. Data Description](#data-description)

Our initial set of variables are as follows. For a reference on data types, check this [diagram](https://o.quizlet.com/8UUywzzaMhY2ZGHrWE7VkA_b.png). 

| Variable      | Description | Data Type |
| ----------- | ----------- | ----------- | 
| Store   | a unique Id for each store  | numerical (discrete)  |
| DayOfWeek   | day of the week (1 = Monday, 7 = Sunday) | numerical (discrete)    |
| Date  | date of each sales entry | date  |
| Sales   | the turnover for any given day (this is what you are predicting) | numerical (continuous)   |
| Customers  | the number of customers on a given day  | numerical (discrete) |
| Open   | an indicator for whether the store was open: 0 = closed, 1 = open | numerical (dummy*)  | 
| Promo  | indicates whether a store is running a promo on that day | numerical (dummy*)   |
| StateHoliday   | indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None | categorical (nominal)  |
| SchoolHoliday  | indicates if the (Store, Date) was affected by the closure of public schools: 1 = affected, 0 = not affected | numerical (dummy*)  |
| StoreType   | differentiates between 4 different store models: a, b, c, d | categorical (nominal)  |
| Assortment | describes an assortment level: a = basic, b = extra, c = extended | categorical (ordinal)  |
| CompetitionDistance   |distance in meters to the nearest competitor store | numerical (continuous)   |
| CompetitionOpenSinceMonth  | gives the approximate month of the time the nearest competitor was opened | numerical (discrete)  |
| CompetitionOpenSinceYear   | gives the approximate year of the time the nearest competitor was opened | numerical (discrete) | 
| Promo2  | promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating | numerical (dummy)  |
| Promo2SinceWeek   |describes the calendar week when the store started participating in Promo2 | numerical (discrete) |
| Promo2SinceYear   |describes the year when the store started participating in Promo2 | numerical (discrete) |
| PromoInterval   |describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store | categorical (nominal) |

*dummy variable is one that takes either 0 or 1. For more details, check [here](https://en.wikipedia.org/wiki/Dummy_variable_(statistics))._


[back to top](#table-of-contents)

---
## 3. Feature Engineering
here we create new features on our dataset to support our modelling. In this task, the business needs are the main compass that guide us for feature creation. There are two tasks within feature engineering:
    *  Start with a hypothesis list, which must be: (1) connected with the main issue/question of the project; (2) do not have clear answers and you expect to get those answers from your dataset. 
    *  Create new variables that will enable you to reject/fail to reject those hypotheses. 
-  **Filtering Variables**: this is the moment to check the business restritions that should be considered in the project. There are two tasks to perform:
    * Data Entry Filtering: is there a specific characteristic that won't be considered on the model? If so, filter entries with such train and exclude them from the dataset.
    * Business Restriction - Feature Selection: if there are some variables that won't be available at the moment of predictions, then it might be wise to drop them from the dataset.

[back to top](#table-of-contents)

---

## 4. Exploratory Data Analysis (EDA)

in this task, our main focus is to explore our dataset and generate valuable insights for the business. . We will divide this task into three parts:
    * Univariate Analysis: check data distribution of each feature and get a first look on how your data is organized as a whole.
    * Bivariate Analysis: check how each feature behaves against the target variable. Here we validate the hypothesis list (created on the Feature Engineering part), generate business insights, and analyze whether to include each variable to the model. 
    * Multivariate Analysis: here we check whether there are repetitive information across features. Since Machine Learning (ML) models work best when data is less complex, dimensionality reduction plays a big role on ML models.

[back to top](#table-of-contents)

---

## 5. Data Preprocessing
data is usually not ordered in a similar manner. Some variables might have an extremely high range, while others might have minimal range. Also, some data might have categorical data on it, and this could be a problem: most ML models perform better when categorical data is transformed to numerical data. Therefore, this task is centered on rescaling and encoding our variables.

[back to top](#table-of-contents)

---

## 6. Feature Selection
In this step, we select the variables that best explain our response (target) variable. There are three methods to do so: 
  * Filter Methods, which is usually the simplest, fastest method to select relevant variables but exclude [multicollinearity issues](https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/); 
  * Embedded Methods, which simply runs a ML algorithm that already has an in-built feature selection step; 
  * Wrapper Methods, which is computationally expensive but a reliable method that uses a ML algorithm to determine the best features to keep.
        
    Detailed explanation on feature selection methods can be found [here](https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/).

[back to top](#table-of-contents)


---
## 7. Machine Learning Modeling
The fun part has just arrived! We divide this step into four tasks:
  
  * **Performance Metrics**: choose suitable metrics to measure performance on our model. In this case, we will use three metrics:
    1. Mean Absolute Error (MAE)
    2. Mean Absolute Percentage Error (MAPE)
    3. Root Mean Squared Error (RMSE) 
  
  * **Modeling:** split the dataset into training and validation data, and choose ML models to get the predictions. To get a criteria for which algorithm to run, check a cheat sheet for machine learning algorithms [here](https://blogs.sas.com/content/subconsciousmusings/2017/04/12/machine-learning-algorithm-use/). 
  For our project, we will run to run the following models:
    1. Mean of the target variable (baseline)
    2. Linear Regression. If performance is poor, the data is probably complex and non-linear.
    3. Random Forest Regressor and XGBoost Regressor

  * **Cross-validation:** in order to measure the real performance of each algorithm, split the dataset into k-folds (usually five folds), choose one fold to be the validation dataset, and run all algorithms. Repeat until you get the results for all folds, and average all results. The algorithm that produces the best result will be used in this prediction project. 

[back to top](#table-of-contents)


---
## 8. Hyperparameter Tuning

in this task, our goal is to find the best parameters which maximizes the learning on our model. There are three methods to find these parameters:
    1. Random Search: this method randomly choose parameters from a given list. It is the fastest method available;
    2. Grid Search: this method is the most complete one and find the absolute best values for each parameter on the model. Once a list of parameters is set, this method performs combination of every single possibility among parameters. Very slow and costly;
    3. Bayesian Search: based on the Bayes' Theorem, this method defines parameters according to prior knowledge. It starts with one initial set of parameters that has its performance calculated. Then, for the next set of parameters, one parameter is changed and its performance is calculated again. If results get better, parameters will be changed. Otherwise, parameters will be kept. This method is faster than Grid Search and slower than Random Search.

[back to top](#table-of-contents)

---
## 9. Error Interpretation & Business Performance

Also known as _Residual Analysis_, in this step we analyze the predictions' performance from a business perspective. Our focus is to translate and interpret errors from our ML model. 

[back to top](#table-of-contents)

---
## 10. Deploy Machine Learning Model to Production



---
## 11. A Sales Predictor Bot



---

## Conclusion



---

## Appendix I - Datasets



---

## Appendix II - References




