# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 22:45:29 2020

@author: malat
"""


import pymysql

conn = pymysql.connect(
    host='database2.cvz9y3wumtvk.us-east-2.rds.amazonaws.com',
    port=int(3306),
    user="admin",
    passwd="akhil12345",
    db="system",
    charset='utf8mb4')


cursor=conn.cursor()

create_table="""
create table salespeople (
salesPersonId int,
salesPersonName varchar(20) 
)

"""

cursor.execute(create_table)
 