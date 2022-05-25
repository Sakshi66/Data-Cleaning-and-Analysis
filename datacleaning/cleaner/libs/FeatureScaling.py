import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FeatureScaling:

	def __init__(self):
		pass

	def fStandardize(self,df,col):
		if col in df:
			dTypeCol = df.dtypes[col]
			if dTypeCol == "int64":
				df[col] = (df[col] - df[col].mean()) / df[col].std()
			else:
				print("Cannot perform feacture encoding on {col} column")
		else:
			print("The column name is invalid!")

		return df

			
	def fNormalize(self,df,col):
		if col in df:
			dTypeCol = df.dtypes[col]
			if dTypeCol == "int64":
				s = (df[col].max() - df[col].min()) 
				df[col] = (df[col] - df[col].min()) / s
			else:
				print("Cannot perform feacture encoding on {col} column")
		else:
			print("The column name is invalid!")

		return df

