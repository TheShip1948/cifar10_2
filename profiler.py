###########################################
# Future Work: 
# ------------- 
# 1. Logging in terminal 
# 2. Logging in file 
# Hint: using interfaces 
###########################################
import time 

class Profiler: 
	
	startTime = 0 
	endTime   = 0
	deltaTime = 0

	def StartTime(self):
		self.startTime = time.time() 

	def EndTime(self):
		self.endTime = time.time() 

	def DeltaTime(self):
		self.deltaTime = time.time() 
		print('Delta time = {} seconds'.format(self.endTime - self.startTime))


