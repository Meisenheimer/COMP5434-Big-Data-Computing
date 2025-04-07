import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


class Engineering():
    def __init__(self):
        pass

    def plot_smooth_dist_by_label(self, data, features, label_col='label', figsize=(18, 15), cols_per_row=3, bw_adjust=1,
                                  filename=""):
        # compute rows
        n_cols = len(features)
        n_rows = (n_cols + cols_per_row - 1) // cols_per_row

        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]

        # draw KDE plot for each feature
        for i, col in enumerate(features):
            ax = axes[i]

            # calculate KDE by label
            for label, color in zip([0, 1], ['#C1E1C1', '#8DB6CD']):
                subset = data[data[label_col] == label][col].dropna()
                sns.kdeplot(
                    data=subset,
                    ax=ax,
                    color=color,
                    label=f'{label_col}={label}',
                    bw_adjust=bw_adjust,
                    linewidth=2,
                    alpha=0.7,
                    fill=True
                )

            ax.set_title(f'{col} Ditribution Classified by Label', fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel('Probability Density', fontsize=10)
            ax.legend(title='Placement')
            ax.grid(True, linestyle='--', alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.savefig(f"figures/Distribution Classified by Label{filename}.png", bbox_inches='tight')
        plt.tight_layout()
        # plt.show()

    def plot_correlation_coefficient(self, df, suffix=""):
        methods = [
            ('pearson', 'Pearson Correlation Coeffiecient'),
            ('spearman', 'Spearman Correlation Coeffiecient'),
            ('kendall', 'Kendall Correlation Coeffiecient')
        ]
        plt.figure(figsize=(18, 6))
        for i, (method, title) in enumerate(methods, 1):
            plt.subplot(1, 3, i)
            corr = df.corr(method=method, numeric_only=True)
            sns.heatmap(corr[['label']].sort_values(by='label', ascending=False),
                        annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(title)

        plt.tight_layout()
        plt.savefig(f"figures/Correlation Analysis{suffix}.png", bbox_inches='tight')
        # plt.show()

    def eda(self, filename: str):
        df = pd.read_csv(filename)
        print(df.head())
        print(df.describe())

        # uniqueness
        is_unique = df['StudentID'].is_unique
        if not is_unique:
            print("Number of duplicated StudentID(s): ", df['StudentID'].duplicated().sum())

        # missing value
        print("Number of missing values:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

        # convert text attribute to 0/1
        df['ExtracurricularActivities'] = df['ExtracurricularActivities'].apply(lambda x: 1 if x == 'Yes' else 0)
        df['PlacementTraining'] = df['PlacementTraining'].apply(lambda x: 1 if x == 'Yes' else 0)
        df[['ExtracurricularActivities', 'PlacementTraining']].head()

        numeric_columns = ['CGPA', 'AptitudeTestScore', 'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        # distribution of numerical columns
        for i, col in enumerate(numeric_columns):
            sns.kdeplot(df[col], ax=axes[i], linewidth=2, fill=True)
            axes[i].set_title(f"Distribution of {col}", fontsize=12)
            axes[i].set_xlabel("Value", fontsize=10)
            axes[i].set_ylabel("Density", fontsize=10)
            axes[i].grid(True, linestyle='--', alpha=0.6)

        if len(numeric_columns) < len(axes):
            for j in range(len(numeric_columns), len(axes)):
                axes[j].set_visible(False)

        plt.savefig("figures/Distribution of Continuous Features.png", bbox_inches='tight')
        plt.tight_layout()
        # plt.show()

        categorical_columns = ['Internships', 'Projects', 'Workshops/Certifications', 'ExtracurricularActivities',
                               'PlacementTraining']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        # countplot of categorical columns
        for i, col in enumerate(categorical_columns):
            if i < len(axes):
                # sns.countplot(data=df, x=col, ax=axes[i], palette='Blues', edgecolor='black')
                sns.countplot(data=df, x=col, ax=axes[i], hue=col, palette='Blues', edgecolor='black', legend=False)
                axes[i].set_title(f"Distribution of {col}", fontsize=12)
                axes[i].set_xlabel(col, fontsize=10)
                axes[i].set_ylabel("Count", fontsize=10)
                axes[i].grid(True, linestyle='--', alpha=0.6, axis='y')

                axes[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                if df[col].nunique() > 10:
                    axes[i].tick_params(axis='x', rotation=45)

        for j in range(len(categorical_columns), len(axes)):
            axes[j].set_visible(False)

        plt.savefig("figures/Distribution of Categorical Features.png", bbox_inches='tight')
        plt.tight_layout()
        # plt.show()

        # check outliers (using box plots)
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(numeric_columns):
            plt.subplot(3, 4, i + 1)
            sns.boxplot(x=df[column])
            plt.title(f"Boxplot of {column}")

        plt.savefig("figures/Boxplot.png", bbox_inches='tight')
        plt.tight_layout()
        # plt.show()

        # plot distribution classified by label to further explore
        features_to_plot = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 'AptitudeTestScore',
                            'SoftSkillsRating', 'ExtracurricularActivities', 'PlacementTraining', 'SSC_Marks',
                            'HSC_Marks']
        self.plot_smooth_dist_by_label(df, features=features_to_plot, bw_adjust=1.2)

        # correlation heatmap
        plt.figure(figsize=(12, 8))
        correlation_matrix = df.drop(['StudentID'], axis=1).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Pearson Correlation Heatmap")
        plt.savefig("figures/Pearson Correlation Heatmap.png", bbox_inches='tight')
        # plt.show()

        self.plot_correlation_coefficient(df)

        print("\nEDA Completed")

        return df

    def one_hot_encode_sklearn(self, df, cols):

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[cols])

        feature_names = []
        for i, col in enumerate(cols):
            categories = encoder.categories_[i]
            feature_names.extend([f"{col}_{cat}" for cat in categories])

        df_encoded = pd.concat([
            df.drop(cols, axis=1),
            pd.DataFrame(encoded_data, columns=feature_names)
        ], axis=1)

        return df_encoded, encoder

    def feature_engineering(self, df, target=False):  # target==True: for training set
        if target:
            df.drop('StudentID', axis=1, inplace=True)
        else:
            df['ExtracurricularActivities'] = df['ExtracurricularActivities'].apply(lambda x: 1 if x == 'Yes' else 0)
            df['PlacementTraining'] = df['PlacementTraining'].apply(lambda x: 1 if x == 'Yes' else 0)

        # new features
        df['Practical Experiences'] = df['Internships'] + df['Projects'] + df['Workshops/Certifications']
        df['I+P'] = df['Internships'] + df['Projects']
        df['I+W'] = df['Internships'] + df['Workshops/Certifications']
        df['P+W'] = df['Projects'] + df['Workshops/Certifications']

        df['Aptitude*SoftSkills'] = df['AptitudeTestScore'] * df['SoftSkillsRating']

        df['Progress between SCC and HSC'] = df['HSC_Marks'] - df['SSC_Marks']

        if target:
            features_to_plot = ['Practical Experiences', 'I+P', 'I+W', 'P+W', 'Aptitude*SoftSkills',
                                'Progress between SCC and HSC']
            self.plot_smooth_dist_by_label(df, features=features_to_plot, figsize=(18, 9), bw_adjust=1.2,
                                      filename=" (new features)")
            self.plot_correlation_coefficient(df, suffix=" (with new features)")

        # df['CGPA less than 7'] = np.where(df['CGPA'] <= 7, 1, 0)
        # df['CGPA more than 8.3'] = np.where(df['CGPA'] >= 8.3, 1, 0)

        df['Internships less than 2'] = np.where(df['Internships'] < 2, 1, 0)
        df.drop('Internships', axis=1, inplace=True)

        df['Workshops/Certifications less than 2'] = np.where(df['Workshops/Certifications'] < 2, 1, 0)
        # df['Workshops/Certifications==2'] = np.where(df['Workshops/Certifications'] == 2, 1, 0)
        df.drop('Workshops/Certifications', axis=1, inplace=True)

        # df['AptitudeTestScore less than 75'] = np.where(df['AptitudeTestScore'] <= 75, 1, 0)
        # df['AptitudeTestScore more than 86'] = np.where(df['AptitudeTestScore'] >= 88, 1, 0)

        # df['SoftSkillsRating less than 4.1'] = np.where(df['SoftSkillsRating'] <= 4.1, 1, 0)
        # df['SoftSkillsRating more than 4.8'] = np.where(df['SoftSkillsRating'] >= 4.8, 1, 0)

        # df['SSC_Marks less than 60'] = np.where(df['SSC_Marks'] <= 65, 1, 0)
        # df['SSC_Marks more than 80'] = np.where(df['SSC_Marks'] >= 80, 1, 0)

        # df['HSC_Marks less than 70'] = np.where(df['HSC_Marks'] <= 70, 1, 0)
        # df['HSC_Marks more than 83'] = np.where(df['HSC_Marks'] >= 83, 1, 0)

        df.drop('Progress between SCC and HSC', axis=1, inplace=True)

        # Select Features
        # selected_features = ['label', 'PlacementTraining', 'CGPA', 'AptitudeTestScore', 'ExtracurricularActivities', 'Workshops/Certifications more than 1',
        #                      'Aptitude*SoftSkills', 'SSC_Marks', 'CGPA less than 7',
        #                      'Workshops/Certifications==2', 'Projects', 'HSC_Marks']

        # selected_features = ['label', 'PlacementTraining', 'CGPA', 'AptitudeTestScore', 'ExtracurricularActivities', 'Workshops/Certifications more than 1',
        #                      'Aptitude*SoftSkills', 'SSC_Marks', 'CGPA less than 7',
        #                      'Workshops/Certifications==2', 'Projects', 'HSC_Marks',
        #                      'I+W', 'I+P', 'P+W',
        #                     'SSC_Marks less than 60', 'SoftSkillsRating', 'SoftSkillsRating less than 4.1']

        # selected_features = ['label', 'PlacementTraining', 'CGPA', 'AptitudeTestScore', 'ExtracurricularActivities', 'Workshops/Certifications more than 1',
        #                      'Aptitude*SoftSkills', 'SSC_Marks',
        #                      'Workshops/Certifications==2', 'Projects', 'HSC_Marks']

        # df = df[selected_features]

        if target:
            df.to_csv("train_new.csv", index=False)
            print("Feature Engineering Completed (Training Set)")
        else:
            df.to_csv("test_new.csv", index=False)
            print("Feature Engineering Completed (Test Set)")

        return df


if __name__ == "__main__":
    work = Engineering()
    # df = work.eda('train.csv')
    # df_train_new = work.feature_engineering(df, target=True)
    # print("\nAfter feature engineering: ")
    # print(df_train_new.head())
    # print(df_train_new.info())
    df = pd.read_csv('test.csv')
    df_test_new = work.feature_engineering(df, target=False)
