import pandas as pd
from flask import render_template
import numpy as np
import plotly.express as px
import plotly
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

class Assignment9:
    @staticmethod
    def binary_map(x):
        return x.map({'Yes': 1, "No": 0})

    @staticmethod
    def process():
        import os
        from pathlib import Path
        filepath = os.path.join(Path(__file__).parent, 'data', '.')

        churn_data = pd.read_csv(f'{filepath}/9_churn_data.csv')
        customer_data = pd.read_csv(f'{filepath}/9_customer_data.csv')
        internet_data = pd.read_csv(f'{filepath}/9_internet_data.csv')

        # merge churn data with customer data
        df_1 = pd.merge(churn_data, customer_data, how='inner', on='customerID')

        # merge with internet usage data
        dataset = pd.merge(df_1, internet_data, how='inner', on='customerID')
        # dataset.isnull().sum()

        # dataset.head()
        # clean the data
        # dataset['TotalCharges'].describe()
        dataset['TotalCharges'] = dataset['TotalCharges'].replace(' ', np.nan)
        dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'])

        value = (dataset['TotalCharges'] / dataset['MonthlyCharges']).median() * dataset['MonthlyCharges']
        dataset['TotalCharges'] = value.where(dataset['TotalCharges'] == np.nan, other=dataset['TotalCharges'])
        # dataset['TotalCharges'].describe()

        varlist = ['PhoneService', 'PaperlessBilling', 'Churn', 'Partner', 'Dependents']
        dataset[varlist] = dataset[varlist].apply(Assignment9.binary_map)
        # dataset.head()

        # one hot encoding and merge
        dummy1 = pd.get_dummies(dataset[['Contract', 'PaymentMethod', 'gender', 'InternetService']], drop_first=True)
        dataset = pd.concat([dataset, dummy1], axis=1)
        # dataset.head()

        # Creating dummy variables for the variable 'MultipleLines'
        ml = pd.get_dummies(dataset['MultipleLines'], prefix='MultipleLines')
        # Dropping MultipleLines_No phone service column
        ml1 = ml.drop(['MultipleLines_No phone service'], 1)
        # Adding the results to the master dataframe
        dataset = pd.concat([dataset, ml1], axis=1)

        # Creating dummy variables for the variable 'OnlineSecurity'.
        os = pd.get_dummies(dataset['OnlineSecurity'], prefix='OnlineSecurity')
        os1 = os.drop(['OnlineSecurity_No internet service'], 1)
        # Adding the results to the master dataframe
        dataset = pd.concat([dataset, os1], axis=1)

        # Creating dummy variables for the variable 'OnlineBackup'.
        ob = pd.get_dummies(dataset['OnlineBackup'], prefix='OnlineBackup')
        ob1 = ob.drop(['OnlineBackup_No internet service'], 1)
        # Adding the results to the master dataframe
        dataset = pd.concat([dataset, ob1], axis=1)

        # Creating dummy variables for the variable 'DeviceProtection'.
        dp = pd.get_dummies(dataset['DeviceProtection'], prefix='DeviceProtection')
        dp1 = dp.drop(['DeviceProtection_No internet service'], 1)
        # Adding the results to the master dataframe
        dataset = pd.concat([dataset, dp1], axis=1)

        # Creating dummy variables for the variable 'TechSupport'.
        ts = pd.get_dummies(dataset['TechSupport'], prefix='TechSupport')
        ts1 = ts.drop(['TechSupport_No internet service'], 1)
        # Adding the results to the master dataframe
        dataset = pd.concat([dataset, ts1], axis=1)

        # Creating dummy variables for the variable 'StreamingTV'.
        st = pd.get_dummies(dataset['StreamingTV'], prefix='StreamingTV')
        st1 = st.drop(['StreamingTV_No internet service'], 1)
        # Adding the results to the master dataframe
        dataset = pd.concat([dataset, st1], axis=1)

        # Creating dummy variables for the variable 'StreamingMovies'.
        smd = pd.get_dummies(dataset['StreamingMovies'], prefix='StreamingMovies')
        smd.drop(['StreamingMovies_No internet service'], 1, inplace=True)
        # Adding the results to the master dataframe
        dataset = pd.concat([dataset, smd], axis=1)
        # dataset.head()

        # drop the columns for which dummies have been created
        dataset = dataset.drop(
            ['Contract', 'PaymentMethod', 'gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
             'OnlineBackup', 'DeviceProtection',
             'TechSupport', 'StreamingTV', 'StreamingMovies'], 1)
        # dataset.head()

        # outliers removal
        # num_telecom = dataset[['tenure', 'MonthlyCharges', 'SeniorCitizen', 'TotalCharges']]
        # num_telecom.describe(percentiles=[.25, .5, .75, .90, .95, .99])
        # dataset.isnull().sum()
        dataset = dataset[~np.isnan(dataset['TotalCharges'])]

        # define feature and target
        X = dataset.drop(['Churn', 'customerID'], axis=1)
        # X.head()

        y = dataset['Churn']
        # y.head()

        # Splitting the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        X_train[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
            X_train[['tenure', 'MonthlyCharges', 'TotalCharges']])

        X_train.head()

        # Model Building
        # Logistic regression model
        import statsmodels.api as sm
        logm1 = sm.GLM(y_train, (sm.add_constant(X_train)), family=sm.families.Binomial())
        logm1.fit().summary()

        # Feature Selection Using RFE
        from sklearn.linear_model import LogisticRegression
        logreg = LogisticRegression()
        # Feature Selection Using RFE
        from sklearn.linear_model import LogisticRegression
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train, y_train)

        # display the coefficients as a dataframe
        feature_cols = X.columns
        coeffs = pd.DataFrame(list(zip(feature_cols, logreg.coef_[0])), columns=['feature', 'coef'])
        coeffs.set_index('feature', inplace=True)
        coeffs.sort_values('coef', ascending=False).head(15)

        # create a bar chart out of it
        fig = px.bar(coeffs.sort_values('coef', ascending=False), height=600)

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Adding a constant
        X_train_sm = sm.add_constant(X_train[feature_cols])
        logm2 = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
        res = logm2.fit()
        res.summary()

        # Getting the predicted values on the train set
        y_train_pred = res.predict(X_train_sm)

        y_train_pred_final = pd.DataFrame({'Churn': y_train.values, 'Churn_Prob': y_train_pred})
        y_train_pred_final['CustID'] = y_train.index
        y_train_pred_final.head()

        # Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
        y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)

        # Let's see the head
        # y_train_pred_final.head()

        # confusion matrix
        from sklearn import metrics
        # confusion_matrix = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted)
        # print(confusion_matrix)

        accuracy_value = metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted)

        # Making predictions on the test set
        X_test[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
            X_test[['tenure', 'MonthlyCharges', 'TotalCharges']])
        X_test = X_test[feature_cols]
        # X_test.head()

        X_test_sm = sm.add_constant(X_test)
        y_test_pred = res.predict(X_test_sm)

        # Converting y_pred to a dataframe which is an array
        y_pred_1 = pd.DataFrame(y_test_pred)
        # y_pred_1.head()

        # Converting y_test to dataframe
        y_test_df = pd.DataFrame(y_test)

        # Putting CustID to index
        y_test_df['CustID'] = y_test_df.index

        y_pred_1.reset_index(drop=True, inplace=True)
        y_test_df.reset_index(drop=True, inplace=True)

        y_pred_final = pd.concat([y_test_df, y_pred_1], axis=1)
        y_pred_final = y_pred_final.reindex(['CustID', 'Churn', 'Churn_Prob'], axis=1)
        y_pred_final['final_predicted'] = y_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.42 else 0)

        baseline_accuracy = metrics.accuracy_score(y_pred_final.Churn, y_pred_final.final_predicted)
        accuracy_improvement = accuracy_value - baseline_accuracy
        values = {
            'accuracy_value': accuracy_value,
            'baseline_accuracy': baseline_accuracy,
            'accuracy_improvement': accuracy_improvement
        }

        return render_template("assignment9.html.j2", graphJSON=graphJSON, values=values)
