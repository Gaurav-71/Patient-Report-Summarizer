import dates
import vsm

def print_files(resDict):
    print("\n\nDocument Name \t\t Date")
    print("--------------------------------------------------")
    for x in resDict:
        print(x,"\t\t",resDict[x])
    print("--------------------------------------------------\n\n")

def search(query):

    vsmResult = vsm.vsm(dates.files, query)

    res = {}

    for rankSet in vsmResult:
        rank = rankSet[0]
        docIndex = rankSet[1] 

        if (rank != 0.0) :
            fileName = dates.files[docIndex]
            fileName = fileName.split(".txt")[0]
            res [fileName] = dates.Dict[fileName]

    print_files(res)
    return res

