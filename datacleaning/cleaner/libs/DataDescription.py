
class DataDescription:

	def __init__(self):
		pass

	def colProperty(self,df,colName):
		tdisc = {}
		if colName in df.columns:
			if df.dtypes[colName] != 'object':
				tdisc["count"] = df[colName].count()
				tdisc["std"] = df[colName].std()
				tdisc["mean"] = df[colName].mean()
				tdisc["min"] = df[colName].min()
				tdisc["max"] = df[colName].max()
				tdisc["null"]=df[colName].isnull().sum()
				coltype = df.dtypes[colName]
				coltype = str(coltype)
				tdisc["datatype"] = coltype

			elif df.dtypes[colName] == 'object':
				tdisc["count"] = df[colName].count()
				#tdisc["unique"] = df[colName].unique()
				#tdisc["frequancy"] = df[colName].value_counts()
				tdisc["null"]=df[colName].isnull().sum()
				coltype = df.dtypes[colName]
				coltype = str(coltype)
				tdisc["datatype"] = coltype
		else:
			print(df.columns)
			print(colName)
			print("Column name is invalid!")

		return(tdisc)

	def allProperty(self, df):
		print(self.df.describe(include='all'))
