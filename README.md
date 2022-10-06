

# OpenStack Loglizer

**OpenStack Loglizer is a machine learning-based log analysis toolkit for automated anomaly detection in OpenStack logs.**

To be able to analyze raw OpenStack logs, we need to parse and structure them first. Identifying anomalous patterns in logs requires an ability to separate the normal space from
 the space in which the anomaly occurs. The proposed model includes four main steps that start from raw
 log parsing followed by different techniques for data preprocessing to prepare them, and finally, the analysis of data and anomaly detection.
 you can see the workflow here:
![Workflow of the proposed model](docs/img/workflow.PNG =250x250)

# Dataset

A dataset of OpenStack logs was required to conduct this study. There is only one open-source dataset of OpenStack, which is not useful and reliable due to several issues in logs
 containing anomalies. Therefore, about 25,000 OpenStack logs are generated along with the fault injection. Each log has an Instance-Id, which shows the VM ID. This ID is unique,
 the total number of unique IDs for VMs are separately presented according to being normal, containing anomalies, and also the time required for generating these logs.

the openstack log dataset is presented [here](https://github.com/ParisaKalaki/openstack-logs).


1. **Log parsing:**

It is not easy to extract useful information from raw logs since they do not have a structured format. Generally, raw logs must first be converted into structured logs in order to
 be analyzed by data mining algorithms. In log parsing, there are many algorithms such as IPLOM, SLCT, Logsig, LKE, Spell, Drain. Using these algorithms, each event template can be
 extracted from unstructured logs, and raw logs can be transformed into properly structured logs. IPLOM algorithm with 87% accuracy resulted in the highest accuracy among other
 algorithms implemented on the OpenStack dataset. For this reason, the present study has used IPLOM for log parsing. 
 ![OpenStack Log after implementing IPLOM algorithm](docs/img/IPLOM.jpg =250x250)
 
2. **Data Preprocessing:**

A number of important and practical features are selected from existing features and also new ones are created.  We categorize the logs into windows by identifying the instance id
 of the virtual machine in OpenStack logs. We categorize all the event templates that have occurred for each ID, and each ID may have a variety of event templates. By the end, a
 matrix showing rows of sessions, and columns of event templates, is created as shown below:
 ![A matrix made up of sessions and event templates](docs/img/instance id.PNG =250x250)
3. **Data Analysis:**

In this step, we discussed how to use PCA, RPCA, ALM, and the data projection onto column space algorithms to propose the PRPCA-CS for data analysis in this research.

4. **Anomaly Detection:**

We train the model and a threshold was determined to detect anomaly.