from flask import render_template
import plotly.express as px
import plotly
import json
from sklearn.model_selection import train_test_split
from sklearn import tree
from customutils import CustomUtils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz
import pydot
import pandas as pd


class Assignment11B:
    @staticmethod
    def process():
        df = CustomUtils.read_file_and_return_df('11b_employee.csv')
        # df.head()

        # pd.set_option("display.float_format", "{:.2f}".format)
        # df.describe()

        df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis="columns", inplace=True)
        # df.head()

        label = LabelEncoder()
        df['Attrition'] = label.fit_transform(df['Attrition'])
        # df.head()

        # create a list of categorical columns, any "object" (str) columns with less than 10 unique values should be fit
        categorical_cols = []
        unique_vals = []
        for column in df.columns:
            if df[column].dtype == object and len(df[column].unique()) <= 10:
                categorical_cols.append(column)
                unique_vals.append(", ".join(df[column].unique()))

        categories = pd.DataFrame.from_dict({
            'Category': categorical_cols,
            'Unique Values': unique_vals
        })
        # categories

        # df.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20));
        categorical_cols.append('Attrition')
        df = df[categorical_cols]
        df.head()
        categorical_cols.remove('Attrition')

        barChartJsons = []
        # plot how every feature correlates with the "target"
        for i, column in enumerate(categorical_cols, 1):
            #     print(df[column].value_counts())
            fig = px.bar(df, x=f'{column}', y='Attrition', height=600, color=f'{column}')
            chartJson = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            barChartJsons.append(chartJson)
            # fig.show()

        conclusions = pd.DataFrame.from_dict({
            'Category': [
                'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
            ],
            'Inference': [
                'The workers who travel rarely are more likely to quit.',
                'The worker in Research & Development are more likely to quit then the workers on other departement.',
                'The workers with Life Sciences and Medical degrees are more likely to quit then employees from other fields of educations.',
                'Male employees are more likely to quit.',
                'The workers in Laboratory Technician, Sales Executive, and Research scientist are more likely to quit the workers in other positions.',
                'Single employees are more likely to quit.',
                'The workers who work more hours are more likely to quit.'
            ],
        })

        # encode all the categorical columns
        label = LabelEncoder()
        for column in categorical_cols:
            df[column] = label.fit_transform(df[column])

        # df.head()

        X = df.drop('Attrition', axis=1)
        y = df.Attrition

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        tree_clf = DecisionTreeClassifier(random_state=42)
        tree_clf.fit(X_train, y_train)

        random_train_scores = Assignment11B.get_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
        # random_test_scores = Assignment11B.get_score(tree_clf, X_train, y_train, X_test, y_test, train=False)

        params = {
            "criterion": ("gini", "entropy"),
            "splitter": ("best", "random"),
            "max_depth": (list(range(1, 20))),
            "min_samples_split": [2, 3, 4],
            "min_samples_leaf": list(range(1, 20)),
        }

        tree_clf = DecisionTreeClassifier(random_state=42)
        tree_cv = GridSearchCV(tree_clf, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=3)
        tree_cv.fit(X_train, y_train)
        best_params = tree_cv.best_params_
        # print(f"Best paramters: {best_params}")

        tree_clf = DecisionTreeClassifier(**best_params)
        tree_clf.fit(X_train, y_train)
        # bestparams_train_score = Assignment11B.get_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
        # bestparams_test_score = Assignment11B.get_score(tree_clf, X_train, y_train, X_test, y_test, train=False)

        features = list(df.columns)
        features.remove("Attrition")

        dot_data = StringIO()
        export_graphviz(tree_clf, out_file=dot_data, feature_names=features, filled=True)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        Image(graph[0].create_png())

        return render_template("assignment11b.html.j2", barChartJsons=barChartJsons,
                               categories=categories.to_html(classes='table table-striped', index=False, justify='center'),
                               conclusions=conclusions.to_html(classes='table table-striped', index=False, justify='center'),

                               random_train_scores=pd.DataFrame.from_dict(random_train_scores).to_html(classes='table table-striped', index=False, justify='center'),
                               # random_test_scores=random_test_scores,

                               # best_params=pd.DataFrame.from_dict(best_params).to_html(classes='table table-striped', index=False, justify='center'),

                               # bestparams_train_score = bestparams_train_score, bestparams_test_score=bestparams_test_score
                               )

    @staticmethod
    def get_score(clf, X_train, y_train, X_test, y_test, train=True):
        if train:
            pred = clf.predict(X_train)
            clf_report = classification_report(y_train, pred, output_dict=True)
            accuracy = f'{accuracy_score(y_train, pred) * 100:.2f}%'
            confusion = f'{confusion_matrix(y_train, pred)}'

            print("Train Result:\n================================================")
            print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
            print("_______________________________________________")
            print(f"CLASSIFICATION REPORT:\n{clf_report}")
            print("_______________________________________________")
            print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")

        elif not train:
            pred = clf.predict(X_test)
            clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
            accuracy = f'{accuracy_score(y_test, pred) * 100:.2f}%'
            confusion = f'{confusion_matrix(y_test, pred)}'

            # print("Test Result:\n================================================")
            # print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
            # print("_______________________________________________")
            # print(f"CLASSIFICATION REPORT:\n{clf_report}")
            # print("_______________________________________________")
            # print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

        return {
            'accuracy_score': accuracy,
            'confusion_matrix': confusion,
            'classification_report': clf_report
        }