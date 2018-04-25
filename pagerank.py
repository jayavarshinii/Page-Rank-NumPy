
# coding: utf-8

# In[226]:


import os 
import math
import numpy
#Implementation of Page rank using the iterative algorithm that uses the power method . The teleportation rate 𝛼=0.15.
def pagerank(doc):
    pages = int(doc.readline())
    links = int(doc.readline())
    #print(pages,links)
    #initalize the adjaceny matrix
    adjmatrix = numpy.zeros((pages, pages))
    for val in doc:
        values=val.split()
        a,b = [int(i) for i in values ]
        adjmatrix[a][b] = 1
    #print(adjmatrix)
    N= int(math.sqrt(adjmatrix.size))
    print("Using initial vector with value of 1/",N,"for each element")
    #Transition matrix
    for a1, row in enumerate(adjmatrix):
        ones = numpy.count_nonzero(row)#calculate the number of ones in every row. 
        for a2, value in enumerate(row):
            if ones:
                adjmatrix[a1][a2] = 0.15/N if value == 0 else (0.85) / ones + (0.15 / N)
            else:
                adjmatrix[a1][a2] = 1/N
    #Calculate the power vector 
    #print(adjmatrix)

    alphamatrix=numpy.array([1/N for i in range(N)])
    #print(alphamatrix)
    while 1:
        rank=numpy.matmul(alphamatrix,adjmatrix)
        #print(rank)
        x1=alphamatrix-rank
        if math.sqrt(sum(i**2 for i in x1)) < 1E-3:
            return rank
        alphamatrix = rank
    return alphamatrix


def main():
    files=os.listdir('testdata/')
    for file in files:
        with open(file) as doc1:
            print("Output for file",file)
            pgrank=pagerank(doc1)
        page = {i: p1 for i, p1 in enumerate(pgrank)}
        page=sorted(page.items(), key=lambda x:x[1])[:10]
        #print(page)
        print("Doc\t Pagerank Score")
        print("========================")
        for documentid, pagerankscore in page:
            print(documentid,"\t", pagerankscore)
        print("========================")


if __name__ == '__main__':
    main()





