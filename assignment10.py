from flask import render_template
from flask import request

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


class Assignment10:
    @staticmethod
    def gender_map(x):
        return x.map({'male': 1, "female": 0})

    @staticmethod
    def process():
        import os
        from pathlib import Path
        filepath = os.path.join(Path(__file__).parent, 'data', '.')

        passengers = pd.read_csv(f'{filepath}/10_titanic.csv')
        feature_cols = ['Pclass', 'Sex', 'Age']
        # passengers.head()

        passengers[['Sex']] = passengers[['Sex']].apply(Assignment10.gender_map)
        # passengers.head()

        # there are some NaN values in age, we use the mean age there
        mean_age = passengers['Age'].mean()
        passengers['Age'].fillna(value=mean_age, inplace=True)

        passengers.head()
        # mean_age

        X = passengers[feature_cols]
        y = passengers['Survived']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=90)
        knn = KNeighborsClassifier(n_neighbors=21)

        knn.fit(X_train, y_train)
        # predict
        y_pred = knn.predict(X_test)
        model_accuracy = metrics.accuracy_score(y_test, y_pred)
        # model_accuracy

        return render_template("assignment10.html.j2", model_accuracy=model_accuracy)

    # @TODO figure out a better way to handle the duplicate code
    @staticmethod
    def predict():
        data = request.form
        age = data.get("age")
        gender = data.get("gender")
        pclass = data.get("pclass")

        # put some default values in case user didnt pass anything
        if gender == "":
            gender = 1

        if pclass == "":
            pclass = 2

        import os
        from pathlib import Path
        filepath = os.path.join(Path(__file__).parent, 'data', '.')

        passengers = pd.read_csv(f'{filepath}/10_titanic.csv')
        feature_cols = ['Pclass', 'Sex', 'Age']
        # passengers.head()

        passengers[['Sex']] = passengers[['Sex']].apply(Assignment10.gender_map)
        # passengers.head()

        # there are some NaN values in age, we use the mean age there
        mean_age = passengers['Age'].mean()
        passengers['Age'].fillna(value=mean_age, inplace=True)

        if age == "":
            age = str(round(mean_age, 2))

        # passengers.head()
        # mean_age

        X = passengers[feature_cols]
        y = passengers['Survived']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=90)
        knn = KNeighborsClassifier(n_neighbors=21)

        knn.fit(X_train, y_train)
        # predict
        y_pred = knn.predict(X_test)

        survived = knn.predict([[pclass, gender, age]])[0]

        survivedString = "Died"
        if survived == 1:
            survivedString = "Survived"

        genderString = "female"
        if gender == "1":
            genderString = "male"

        pclassString = "Third"
        if pclass == "1":
            pclassString = "First"
        elif pclass == "2":
            pclassString = "Second"

        return f'a person with <b>{genderString}</b> gender of <b>{age}</b> age in <b>{pclassString}</b> class would ' \
               f'have <b>{survivedString}</b> according to knn '
