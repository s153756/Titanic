from typing import Tuple
import pandas as pd


class TitanicPreprocessor:
    def load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep="\t")
        int_cols = ["Survived", "PassengerId", "Pclass", "SibSp", "Parch"]
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        return df

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        y = df['Survived']
        X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
        
        X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
        X['Age'] = X['Age'].fillna(X['Age'].median())
        X['Fare'] = X['Fare'].fillna(X['Fare'].median())
        
        most_frequent_port = X['Embarked'].value_counts().idxmax()
        X['Embarked'] = X['Embarked'].fillna(most_frequent_port)
        embarked_dummies = pd.get_dummies(X['Embarked'], prefix='Embarked', drop_first=False)
        X = pd.concat([X.drop('Embarked', axis=1), embarked_dummies], axis=1)
        
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        X['CabinPresent'] = (df['Cabin'].notna()).astype(int)
        
        X['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.')[0]
        common_titles = ['Mr', 'Mrs', 'Miss', 'Master']
        X['Title'] = X['Title'].apply(lambda x: x if x in common_titles else 'Rare')
        title_dummies = pd.get_dummies(X['Title'], prefix='Title', drop_first=False)
        X = pd.concat([X.drop('Title', axis=1), title_dummies], axis=1)
        
        return X, y
