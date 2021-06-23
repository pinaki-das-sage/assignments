from flask import render_template
import plotly.express as px
import plotly
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn import tree
from customutils import CustomUtils


class Assignment11:
    @staticmethod
    def process():
        bank = CustomUtils.read_file_and_return_df('11a_bank.csv')
        # bank.head()
        bank_data = bank.copy()

        # Combine similar jobs into categiroes
        bank_data['job'] = bank_data['job'].replace(['admin.'], 'management')
        bank_data['job'] = bank_data['job'].replace(['housemaid'], 'services')
        bank_data['job'] = bank_data['job'].replace(['self-employed'], 'entrepreneur')
        bank_data['job'] = bank_data['job'].replace(['retired', 'student', 'unemployed', 'unknown'], 'others')

        # Combine 'unknown' and 'other' as 'other' isn't really match with either 'success' or 'failure'
        bank_data['poutcome'] = bank_data['poutcome'].replace(['other'], 'unknown')
        bank_data.poutcome.value_counts()

        # data cleanup
        bank_data.drop('contact', axis=1, inplace=True)

        bank_data['default_cat'] = bank_data['default'].map({'yes': 1, 'no': 0})
        bank_data.drop('default', axis=1, inplace=True)

        bank_data["housing_cat"] = bank_data['housing'].map({'yes': 1, 'no': 0})
        bank_data.drop('housing', axis=1, inplace=True)

        bank_data["loan_cat"] = bank_data['loan'].map({'yes': 1, 'no': 0})
        bank_data.drop('loan', axis=1, inplace=True)

        bank_data.drop('month', axis=1, inplace=True)
        bank_data.drop('day', axis=1, inplace=True)

        bank_data["deposit_cat"] = bank_data['deposit'].map({'yes': 1, 'no': 0})
        bank_data.drop('deposit', axis=1, inplace=True)

        bank_with_dummies = pd.get_dummies(data=bank_data, columns=['job', 'marital', 'education', 'poutcome'], \
                                           prefix=['job', 'marital', 'education', 'poutcome'])
        # bank_with_dummies.head()
        fig = px.bar(bank_data, x='job', y='deposit_cat', height=600, color='job')
        barchartJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # make a copy
        bankcl = bank_with_dummies
        # The Correltion matrix
        corr = bankcl.corr()
        # corr

        # Train-Test split: 20% test data
        data_drop_deposite = bankcl.drop('deposit_cat', 1)
        label = bankcl.deposit_cat
        data_train, data_test, label_train, label_test = train_test_split(data_drop_deposite, label, test_size=0.2,
                                                                          random_state=50)

        # Decision tree with depth = 2
        dt2 = tree.DecisionTreeClassifier(random_state=1, max_depth=2)
        dt2.fit(data_train, label_train)
        dt2_score_train = dt2.score(data_train, label_train)
        dt2_score_test = dt2.score(data_test, label_test)

        # Decision tree with depth = 3
        dt3 = tree.DecisionTreeClassifier(random_state=1, max_depth=3)
        dt3.fit(data_train, label_train)
        dt3_score_train = dt3.score(data_train, label_train)
        dt3_score_test = dt3.score(data_test, label_test)

        # Decision tree with depth = 4
        dt4 = tree.DecisionTreeClassifier(random_state=1, max_depth=4)
        dt4.fit(data_train, label_train)
        dt4_score_train = dt4.score(data_train, label_train)
        dt4_score_test = dt4.score(data_test, label_test)

        # Decision tree with depth = 6
        dt6 = tree.DecisionTreeClassifier(random_state=1, max_depth=6)
        dt6.fit(data_train, label_train)
        dt6_score_train = dt6.score(data_train, label_train)
        dt6_score_test = dt6.score(data_test, label_test)

        # Decision tree: To the full depth
        dt1 = tree.DecisionTreeClassifier()
        dt1.fit(data_train, label_train)
        dt1_score_train = dt1.score(data_train, label_train)
        # print("Training score: ", dt1_score_train)
        dt1_score_test = dt1.score(data_test, label_test)
        # print("Testing score: ", dt1_score_test)

        # convert all data to pandas df and sent to template to print
        scores = {
            "Tree Depth": ["2", "3", "4", "6", "max"],
            "Training score": [dt2_score_train, dt3_score_train, dt4_score_train, dt6_score_train, dt1_score_train],
            "Testing score": [dt2_score_test, dt3_score_test, dt4_score_test, dt6_score_test, dt1_score_test]
        }
        scoresDf = pd.DataFrame.from_dict(scores)
        scoresDfHTML = scoresDf.to_html(classes='table table-striped', index=False, justify='center')

        # Extract the deposte_cat column (the dependent variable)
        # corr_deposite = pd.DataFrame(corr['deposit_cat'].drop('deposit_cat'))
        # corr_deposite.sort_values(by='deposit_cat', ascending=False)

        tree2 = CustomUtils.get_base64_encoded_image(dt2, data_train.columns)
        tree3 = CustomUtils.get_base64_encoded_image(dt3, data_train.columns)

        return render_template("assignment11.html.j2", barchartJSON=barchartJSON,
                               scoresDfHTML=scoresDfHTML,
                               tree2=tree2, tree3=tree3)
