# User-Centric Determination of Malicious Insiders at Granular Level Using Machine Learning

## Abstract

Malicious insider attacks are seldom considered while network system security issues are discussed
and thus remain a challenge that is yet to be mastered. A vast number of organizations and
government agencies would benefit from a state-of-the-art insider threat detection system.
An ML-based system that scrutinizes the levels of data granularity, with limited onshore monitoring,
on different training settings is proposed. Four different algorithms - Logistic Regression (LR),
Neural Network (NN), Random Forest (RF), and XGBoost (XG) are utilized. A minute analysis of
common insider threat cases is offered, along with various performance measures, to aid in the fair
assessment of system stability.

The proposed model is trained and validated using the Insider Threat Test Dataset synthesized by the
researchers at Carnegie Mellon University.

**KEYWORDS**: Malicious insider, Data granularity, Insider Threat Test Dataset.

## SUMMARY OF THE BASE PAPER

### Base Paper Details:

**Title:** Analyzing Data Granularity Levels for Insider Threat Detection using Machine Learning

**Journal Name:** IEEE Transactions on Network and Service Management

**Publisher:** IEEE

**Year:** 2020

**Paper Index Type:** SCIE

### Introduction:

Insider threats are security issues caused by employees or trusted partners that have legitimate information or data that can cause serious damage to an organization’s network by leaking sensitive information intentionally. Here we propose a model which can detect such insiders based on their transaction history and behavioral changes.
We have used synthesized data by CERT and trained ML algorithms (LR, NN, RF, XG) and compared their performances on how well they generalize to new users in identifying their malicious intent

### Novelty of The Base Paper:

The paper distinctly provides for a user-centered insider threat detection system, in which different granular levels are taken into consideration under various training conditions for data analysis.
The system additionally also gives the analyst more detailed information on user behavior.

### Research Addressed:

A significant interest of the research conducted in the paper involves examining the proficiency of ML-based methods for insider threat detection in corporate and organizational network settings.
Specific interest involves the evaluation of the ML-based methods trained with a restricted set of labeled data to detect unknown malicious users and finding out the capability of the learned algorithm to generalize the detection of unknown malicious insider users.
In addition, distinction between malicious actions and users detected is addressed, meaning that the variety of user roles within an organization that could affect the number / types of actions performed by the user, classifying them to be either normal or malicious, is also taken into account. 

### Proposed Solution:

The system proposed for detecting malign user behavior and insider threats is described in steps below: 
1) Data collection: Required data is collected from various sources and is stored in appropriate formats. The sources under consideration are: 
<ul>
	<li> User activities such as HTTP traffic, emails sent, device activity, etc., </li>
	<li> Organization framework and user profile details. </li>
</ul>
2) Data pre-processing: Processing of the collected data to synthesize numerical feature vectors.
3) Deployment of ML algorithms with the constructed feature vectors as input. 
4) Presentation of results in various formats, and detailed analysis.

## DETAILED ARCHITECTURE:

### Data Preprocessing:

The dataset used in this work is the release 5.2 of the CERT dataset which consists of the data of an organization having 2000 employees over the time period of 1.5 years. This data set contains user actions classified as: login (or) logoff, HTTP, email, file, psychometric and device, along with organizational framework and user profile details.
The detected malign user here, is involved in either of the following scenarios: data exfiltration (1), intellectual property theft (2, 4) to IT sabotage (3).

For data preprocessing, feature extraction is performed at different granularity levels.The different granular levels under consideration are briefed below:

| Data type | Description |
| ---- | ---- |
| User-Week | User actions on all PCs, per week|
| User-Day | User actions on all PCs, per day |
| User-Session | User actions, per session i.e., from login to log off, on a PC |
| User-Subsession T | i hours of user actions, per session |
| User-Subsession N | j user actions, per session |

For our work, we extract two kinds of features:
1. Frequency-based features: The number of various kinds of user actions during the aggregated time period.
2. Statistical features: Statistics describing data including mean, median and standard deviation.
The numerical vectors obtained after the feature extraction aid in training the 4 ML techniques.

### ML Algorithms:

<ul>
	#### <li> Logistic Regression:
		In this work, σ, the logistic function with l2 regularization is used to model the probability of normal and  malicious insider behavior for each input x. </li>
	<li> #### Neural Network:
		The NN used in this work uses rectified linear activation functions in a multilayer perceptron with a maximum of 3 hidden layers.
		Adam formulation of stochastic gradient descent drives the backpropagation in this work and rectified linear activation functions are used in the hidden layers. </li>
	<li> #### Random Forest:
		The properties of RF make it a robust algorithm which is leveraged for insider threat detection in this work. </li>
	<li> #### XGBoost:
		XGBoost provides for certain improvements over traditional gradient boosting techniques and hence provides for a scalable tree boosting system which is in turn used to reduce delay in malicious insider detection. </li>
</ul>
	
### Experiment Settings:

Two experiments are carried out in this work.
The first experiment is aimed to distinguish between traditional ML applications and cyber security environments simulating the real world. A comparison between the realistic setting and an idealistic (traditional) setting is carried out for this purpose.
**Realistic:** Labeled data is acquired from a limited set of employees over a 50% of the time period.
**Idealistic:** From the complete data set, a random 50% of the data is used.

The second experiment is aimed to explore the data’s granularity. Therefore, both instance-based and user-based results are analyzed in a realistic setting on 7 different data granularity levels.

## LITERATURE SURVEY:

### Existing Techniques For Insider Threat Detection:

| S. No | Author  | Methodology Proposed |
| ---- | ---- | ---- |
| 1 | Eberle et al. | Graph based anomaly detection system |
| 2 | Caputo et al. | Monitoring users using Bayesian Networks |
| 3 | Parveen et al. | Incremental learning approach based on streaming data |
| 4 | Rashid et al. | Hidden Markov Model |
| 5 | Eldardiry et al. | Hybrid combination of anomaly detectors |
| 6 | Salem et al. | Anomaly based detection of masqueraders |
| 7 | Gavai et al. | Different ML methods |
| 8 | Tuor et al. | Anomaly detection based on deep neural network |
| 9 | Bose et al. | Scalable ML algorithms on a fusion of heterogeneous data streams |
| 10 | Le et al. | Genetic programming approach |

### Results of Other Works:

| Method | Results |
| ---- | ---- |
| Bayesian Network models | Instance-based Recall = 100%, Precision =  29.8% |
| Hidden Markov Models | Instance-based AUC = 0.83 |
| Supervised quitter detection and unsupervised insider threat detection| Instance-based AUC = 0.77 |
| Unsupervised deep learning (RNN and LSTM) | Instance-based recall upto 35.7% at 1000 alerts daily |
| Streaming anomaly detection algorithms | Instance-based recall = 50%, Precision = 8% |
| Ensemble of anomaly detection approaches | Instance-based AUC upto 0.97 on monthly data |

### Merits of the Existing Techniques:

Some of the above mentioned methods have better recall and AUC values when compared to some of the algorithms used in this work.

### Demerits of the Existing Techniques:

1. Inability to adjust or improve previous results while working with a more limited portion of user data. 
2. Several intricate analyses of user-based results and internal threat scenarios are carried out to gain better understanding of the detection performance of the system.

## Code:


