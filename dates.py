import os
import datetime

files = os.listdir("./static/Data")
print(files)
dates = {
    'Navarro Del Rio.txt' : datetime.datetime(1996, 3, 13),
    'Jesse Pinkman.txt' : datetime.datetime(2000, 4, 20), 
    'Darlene Jackson.txt' : datetime.datetime(2007, 8, 25),
    'Chadwick Boseman.txt' : datetime.datetime(2005, 3, 7),
    'Maya Miller.txt' : datetime.datetime(2003, 3, 25), 
    'Wyatt Langmore.txt' : datetime.datetime(1997, 2, 16), 
    'Wendy Byrde.txt' : datetime.datetime(1996, 2, 21), 
    'Phoebe Buffay.txt' : datetime.datetime(2009, 6, 21), 
    'Chandler Bing.txt' : datetime.datetime(1999, 8, 2), 
    'Robert Downey Jr..txt' : datetime.datetime(2003, 8, 1),
    'Jacob Snell.txt' : datetime.datetime(2005, 9, 1), 
    'Chris Evans.txt' : datetime.datetime(2007, 1, 30), 
    'Helen Pierce.txt' : datetime.datetime(2002, 11, 25), 
    'Rachel Green.txt' : datetime.datetime(2014, 2, 25), 
    'Jennifer Winged.txt' : datetime.datetime(1995, 8, 25), 
    'Scarlet Johanson.txt' : datetime.datetime(1999, 1, 2), 
    'Marsha Smith.txt' : datetime.datetime(2004, 9, 25), 
    'Samuel L Jackson.txt' : datetime.datetime(2010, 13, 5), 
    'Sebastian Stan.txt' : datetime.datetime(2011, 11, 25)
    }


