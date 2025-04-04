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

Your goal is to train a model to predict (classify) whether each student in another test dataset will be placed (i.e., can find a job) based on the training set. Evaluation is based on the Macro-F1 metric. 将Macro-F1优化到0.85以上甚至0.95以上

Number of samples in the training set: 7093

Number of students in another test set: 2907

Hints:

- Note that the training data may be dirty
- You may consider the following process: exploration(preliminary analysis such as visualization), data processing(data cleaning, feature extraction and feature scaling), 建模调参并对比模型Macro-F1表现（结果部分输出并可视化）, insights and actions based on the results

本数据集中需要关注的影响Macro-F1表现的因素：

- **label类别不平衡**
- 特征的预测能力，除了独热编码，是否有必要构造新特征，计算重要特征，剔除相关度低和低贡献特征
- 模型选择与复杂度，参数调优
- 如果训练集（7093 样本）和测试集（2907 样本）的特征分布或类别比例差异较大，模型在测试集上的 Macro-F1 会下降
- 样本数量与多样性，检查特征组合的覆盖率？

Sample Submission: sample_submission.csv

- This file is a sample of your submission about the predictions of test.csv.
- This file includes two columns.
o StudentID: the identifier of each record in test.csv.
o label: the prediction of this record.



You can use low-level third-party packages to facilitate your implementation.

Your implementation should involve sufficient technical details developed by
yourselves.
	o DO NOT simply call ready-to-use classification models provided in existing packages, as a Blackbox, to finish the project. 