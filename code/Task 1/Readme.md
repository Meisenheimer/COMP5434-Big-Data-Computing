This dataset records academic and training performance of some students. 

Training data includes training records with ground-truth class labels. Each row is a data record with its StudentID, 10 attributes, and a label.

- StudentID: Unique identifier of each record.
- CGPA: It is the overall grades achieved by the student.
- Projects: Number of projects a student has done.
- Workshops/Certifications: As there are multiple courses available online student opt for them to upskill themselves.
- AptitudeTestScore: Aptitude test are generally a part of the recruitment process to understand the Quant and logical thinking of the student.
- SoftSkillsRating: Communication is a key role that plays in the placement or in any aspect of the life.
- ExtracurricularActivities: This helps provide and insight about the personality of an individual regarding how much he/she is active other than the academic.
- PlacementTraining: It is provided to students in college to ace the placement process.
- SSC_Marks: Senior Secondary Marks.
- HSC_Marks: Higher Secondary Marks.

- label: It is our classification target, indicating whether this student is placed (1 indicates placed.)

The goal is to train a model to predict (classify) whether each student in another test dataset will be placed (i.e., can find a job) based on the training set. Evaluation is based on the Macro-F1 metric.

Number of samples in the training set: 7093

Number of students in another test set: 2907

Hints:

- Note that the training data may be dirty
- Consider the following process: exploration (preliminary analysis such as visualization), data processing (data cleaning, feature extraction and feature scaling), modeling, tuning, comparing Macro-F1 (results visualization), insights based on the results

Some factors that need to be focused：

- Label category imbalance
- Is it necessary to construct new features, calculate important features, and eliminate low correlation and low contribution features for the predictive ability of features, in addition to single hot encoding
- Model selection and complexity, parameter tuning
- If there is a significant difference in the feature distribution or category ratio between the training set (7093 samples) and the test set (2907 samples), the Macro-F1 of the model on the test set will decreas
- Sample size and diversity, check the coverage of feature combinations?

Sample Submission: sample_submission.csv

- This file is a sample of your submission about the predictions of test.csv.
- This file includes two columns.
o StudentID: the identifier of each record in test.csv.
o label: the prediction of this record.
