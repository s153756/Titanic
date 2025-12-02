This module provides a TitanicPreprocessor class used to load and transform the Titanic dataset into a model-ready format.
It handles reading the data, cleaning missing values, encoding categorical features,
and creating several engineered features often used in Titanic survival prediction tasks

##USAGE:

from titanic_preprocessor import TitanicPreprocessor

tp = TitanicPreprocessor()
df = tp.load_data("titanic.tsv")
X, y = tp.transform(df)
