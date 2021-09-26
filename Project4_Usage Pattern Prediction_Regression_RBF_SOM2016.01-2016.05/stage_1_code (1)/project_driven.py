from dataBase import *
from dataExtrc import *
user=dataBase()
user.welcome()
user.preDB()
dataext=dataExtrc()
result=user.openDB(dataext)

