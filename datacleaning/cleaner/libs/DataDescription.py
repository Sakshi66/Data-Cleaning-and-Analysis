
class DataDescription:
	df = None

	def __init__(self):
		pass

	def colProperty(self):
		for col in self.df.columns:
			print(col)
		print("")

		while(True):
			colName = input("Enter column name: ")
			print("")
			if colName in self.df.columns:
				if self.df.dtypes[colName] != 'object':
					print("count  ",self.df[colName].count())
					print("std  ",self.df[colName].std())
					print("mean ",self.df[colName].mean())
					print("min  ",self.df[colName].min())
					print("max  ",self.df[colName].max())
				elif self.df.dtypes[colName] == 'object':
					print("count  ",self.df[colName].count())
					print("unique  ",self.df[colName].unique())
					print("frequancy\n",self.df[colName].value_counts())

				break
			else:
				print("Column name is invalid!")

	def allProperty(self):
		print(self.df.describe(include='all'))

	def showDataset(self):
		n = 1
		while(True):
			try:
				val = input("How many rows to print(>0): ")
				n = int(val)
				break;
			except:
				print(f"{val} is not a valid option!")

		print(self.df.head(n))