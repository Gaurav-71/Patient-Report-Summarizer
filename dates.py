import os
import datetime

files = os.listdir("./static/Data")

try:
    files.remove('.DS_Store')
except:
    pass

Dict = {
    'Navarro Del Rio': datetime.datetime(1996, 3, 13),
    'Jesse Pinkman': datetime.datetime(2000, 4, 20),
    'Darlene Jackson': datetime.datetime(2007, 8, 25),
    'Chadwick Boseman': datetime.datetime(2005, 3, 7),
    'Maya Miller': datetime.datetime(2003, 3, 25),
    'Wyatt Langmore': datetime.datetime(1997, 2, 16),
    'Wendy Byrde': datetime.datetime(1996, 2, 21),
    'Phoebe Buffay': datetime.datetime(2009, 6, 21),
    'Chandler Bing': datetime.datetime(1999, 8, 2),
    'Robert Downey Jr': datetime.datetime(2003, 8, 1),
    'Jacob Snell': datetime.datetime(2005, 9, 1),
    'Chris Evans': datetime.datetime(2007, 1, 30),
    'Helen Pierce': datetime.datetime(2002, 11, 25),
    'Rachel Green': datetime.datetime(2014, 2, 25),
    'Jennifer Winged': datetime.datetime(1995, 8, 25),
    'Scarlet Johanson': datetime.datetime(1999, 1, 2),
    'Marsha Smith': datetime.datetime(2004, 9, 25),
    'Samuel L Jackson': datetime.datetime(2010, 5, 13),
    'Sebastian Stan': datetime.datetime(2011, 11, 25)
}

datesDict = {}


def addValues(key, value):
    datesDict[key] = value


[addValues(key, value)
 for (key, value) in sorted(Dict.items(),  key=lambda x: x[1])]
