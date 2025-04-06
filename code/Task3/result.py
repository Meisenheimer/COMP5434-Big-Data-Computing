import os
import re
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()
    folder_list = os.listdir(args.dir)
    for folder in folder_list:
        if (not os.path.isdir(os.path.join(args.dir, folder))):
            continue
        with open(os.path.join(args.dir, folder, "log.txt"), "r") as fp:
            text = fp.readlines()
        model = re.search(r"model='.+?',", text[0], re.M | re.I).group(0)[7:-2]
        print(folder)
        if (model.lower() == "logistic"):
            alpha_1 = re.search(r"alpha_1=.+?,", text[0], re.M | re.I).group(0)
            alpha_2 = re.search(r"alpha_2=.+?,", text[0], re.M | re.I).group(0)
            degree = re.search(r"degree=.+?,", text[0], re.M | re.I).group(0)
            feat_selection = re.search(r"feat_selection=.+?,", text[0], re.M | re.I).group(0)
            print(f"{model},\t{alpha_1}\t{alpha_2}\t{degree}\t{feat_selection}\t{text[1].split(',')[1]}\t{text[2].strip()}\t{text[3].strip()}")
        elif (model.lower() == "mlp"):
            dropout = re.search(r"dropout=.+?,", text[0], re.M | re.I).group(0)
            hidden_size = re.search(r"hidden_size=.+?,", text[0], re.M | re.I).group(0)
            feat_selection = re.search(r"feat_selection=.+?,", text[0], re.M | re.I).group(0)
            print(f"{model},\t{dropout}\t{hidden_size}\t{feat_selection}\t{text[1].split(',')[1]}\t{text[2].strip()}\t{text[3].strip()}")
        # exit(0)
