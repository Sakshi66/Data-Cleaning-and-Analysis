from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

class Encoding:

	def __init__(self):
	    pass

	def showCatCols(self):
		self.df_cat = df.select_dtypes(include =['object'])
		df_cat_unique = self.df_cat.nunique().to_frame().reset_index()
		df_cat_unique.columns = ['Variable','DistinctCount']
		print(df_cat_unique)

	def ordinalEncoding(self):
		pass

	def oneHotEncoding(self,df,colName):
		if colName in df:
			df_cat = df.select_dtypes(include =['object'])
			if comName in df_cat:				
				encoded_df = pd.get_dummies(df[[colName]])
				# This returns a new dataframe with a column for every unique value that exists,
				# along with either a 1 or 0 specifying the presence of that rating for a
				# given observation.

				# we want this to be part of the original dataframe. In this case, we attach our
				# new dummy coded frame onto the original frame using "column-binding.
				# pd.concat([df, encoded_df], axis=1)
				df = pd.concat([df, encoded_df], axis=1)

				# Dropping the original column
				df = df.drop([colName], axis=1)
			
			else:
				print("Unable")
		else:
			print("The column name is invalid!")

		return df