import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

class processdata():#must make manual changes!
	def __init__(self,data):
		self.df = pd.read_csv(data)
		#can manually add more dfs if you have multiple excel files

	def printdata(self):
		print(self.df)

	@staticmethod
	def fitToEquation(X,Y,deg,p0):
		def func(x,m,c):
			return m*x + c

		p , cov =scipy.optimize.curve_fit(func,X,Y, p0)

		return p , np.sqrt(np.diag(cov)) 

	def plotlogV(self,Xname,Yname,Xlabel,Ylabel, start = 0, end = -1,p0 = [1,1]):

		X = np.log(self.df[Xname].dropna())[start:end]

		Y = (self.df[Yname].dropna() + min(self.df[Yname].dropna()))[start:end]

		p , errors = self.fitToEquation(Y,X,1,p0)
		pfit = np.poly1d(p)

		print(f"{p} are the coefficients of X in the modelled equation starting from the highest power")
		print(f"{errors} are the errors in the coefficients")

		plt.plot(Y,X,'.',  markersize = 10)
		plt.plot(Y,pfit(Y))
		plt.legend(["data","fit"])
		plt.xlabel(Xlabel)
		plt.ylabel(Ylabel)
		plt.show()

	def plotVt(self,Xname,Yname,Xlabel,Ylabel, start = 0, end= -1):

		X = self.df[Xname].dropna()[start:end]
		Y = self.df[Yname].dropna()[start:end]

		plt.plot(X, Y,'.',  markersize = 10)
		plt.xlabel(Xlabel)
		plt.ylabel(Ylabel)
		plt.show()


if __name__ == "__main__":
	processor = processdata("SQR01.CSV")
	processor.plotlogV("C1 in V","in s","Log of Potential Difference (ln(V))","Time (s)", 256)
	processor.plotVt("in s","C1 in V","Time (s)","Potential Difference (V)")
	processor = processdata("SQR02.CSV")
	processor.plotVt("in s","C2 in V","Time (s)","Potential Difference (V)",5000,80000)
	processor.plotlogV("C2 in V","in s","Log of Potential Difference (ln(V))","Time (s)", 5000, 60000 , [-0.000057,-0.000475])