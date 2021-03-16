import pyodbc
import datetime
import time
server = '10.194.2.45, 1433'
db1 = 'Deepface'
uname = 'dk'
pword = 'dkadmin'
tcon= 'No'
try:
    cnxn = pyodbc.connect(driver='{SQL Server}', host=server, database=db1,trusted_connection=tcon, user=uname, password=pword)
    cursor = cnxn.cursor()
    print("connection to database suceeded")
except:
    print("connection failed")