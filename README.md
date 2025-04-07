# Big-Data-Computing

**If there is any problem, please email me at [zeyu-asparagine.wang@connect.polyu.hk](mailto:zeyu-asparagine.wang@connect.polyu.hk) .**

Requirement:

- `Python`: All the packages needed are commonly used (e.g. torch, scikit-learn, numpy, matplotlib, tqdm).

The followings are description for each folder:

- `./codes/`: The codes and data for the project, there are three folders inside, each folder for one task;
- `./report/`: The final report;
- `./slide/`: The slide for presentation.

## In ./codes/:

the `analysis.py` is used to generate the figures used in the report and slide, with the output folder `Analysis`

the `.sh` file includes the command to run the code, where the `run.sh` includes the command corresponding to the submission on kaggle, and others for ablation test.

the `result` and `result-{method}` folder includes the result output by the code, the `log.txt` record the option, f1 score and time cost and `mem.txt` is the memory cost. The results in `result` are submitted to the kaggle, and others are ablation test.

