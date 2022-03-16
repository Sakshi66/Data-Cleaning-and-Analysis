import numpy as nu
class NullValues:

	def __init__(self):
		pass

	def rmNullCol(self,df,colName):
		if colName in df:
			df = df.drop([colName], axis=1)
		else:
			print("The column name is invalid!")
		
		return df

	def fillMean(self,df,colName):
		if colName in self.df.columns:
			if self.df.dtypes[colName] != 'object':
				mean_value = self.df[colName].mean()
				self.df[colName].fillna(value=mean_value, inplace=True)
			else:
				print("Cannot Replace with Mean")
		else:
			print("invalid column name")

		return df

	def fillMode(self,df,colName):
		if colName in self.df.columns:
			mode_value = df[colName].mode()
			df[colName].fillna(value=mode_value, inplace=True)
		else:
			print("The column name is invalid!")

		return df
			
	def fillMedian(self,df,colName):
		if colName in self.df.columns:
			if df.dtypes[colName] != 'object':
				median_value = df[colName].median()
				df[colName].fillna(value=median_value, inplace=True)
			elif df.dtypes[colName] == 'object':
				print("Cannot replace with Median")
		else:
			print("The column name is invalid!")

		return df
			







