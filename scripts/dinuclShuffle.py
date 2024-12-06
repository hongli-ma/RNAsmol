### altschulEriksonDinuclShuffle.py
### P. Clote, Oct 2003
### dinuclShuffle function 

import copy, random, sys
# MEME Suite libraries
sys.path.append('anaconda3/envs/bioinfo_py36/lib/meme-5.0.5/python')
import alphabet_py3 as alphabet
import sequence_py3 as sequence


def computeCounts(s, alph):
    alen = alph.getFullLen()
    nuclCnt = [0 for i in range(alen)]
    dinuclCnt = [[0 for i in range(alen)] for j in range(alen)]
    # compute counts
    if len(s) > 0:
        y = alph.getIndex(s[0])
        nuclCnt[y] = 1
        for i in range(len(s) - 1):
            x = y
            y = alph.getIndex(s[i + 1])
            nuclCnt[y] += 1
            dinuclCnt[x][y] += 1
    return nuclCnt, dinuclCnt

def chooseEdge(x, dinuclCnt, alph):
    z = random.random()
    denom = sum(dinuclCnt[x])
    numerator = 0
    for y in range(len(dinuclCnt[x]) - 1):
        numerator += dinuclCnt[x][y]
        if z < float(numerator)/float(denom):
            dinuclCnt[x][y] -= 1
            return y
    y = len(dinuclCnt[x]) - 1
    dinuclCnt[x][y] -= 1
    return y

def connectedToLast(edgeList, usedSymI, lastSymI):
    D = {}
    for x in usedSymI:
        D[x] = 0
    for a, b in edgeList:
        if b == lastSymI:
            D[a] = 1
    for i in range(len(usedSymI) - 1):
        for a, b in edgeList:
            if D[b] == 1:
                D[a] = 1
    for x in usedSymI:
        if x != lastSymI and D[x] == 0:
            return False
    return True

def eulerian(s, alph):
    nuclCnt, dinuclCnt = computeCounts(s, alph)
    #compute nucleotides appearing in s
    usedSymI = [x for x in range(len(nuclCnt)) if nuclCnt[x] > 0]
    #create dinucleotide shuffle L
    lastSymI = alph.getIndex(s[-1])
    ok = False
    while not ok:
        counts = copy.deepcopy(dinuclCnt)
        edgeList = []
        for x in usedSymI:
            if x != lastSymI:
                edgeList.append( (x, chooseEdge(x, counts, alph)) )
        ok = connectedToLast(edgeList, usedSymI, lastSymI)
    return edgeList

def computeList(s, alph):
    out = [[] for i in range(alph.getFullLen())]
    if len(s) > 0:
        y = alph.getIndex(s[0])
        for i in range(len(s) - 1):
            x = y
            y = alph.getIndex(s[i + 1])
            out[x].append(y)
    return out

def dinuclShuffle(s, alph = alphabet.dna()):
    # check we can actually shuffle it
    if len(s) <= 2:
        return s
    # determine how to end the sequence
    edgeList = eulerian(s, alph)
    # turn the sequence into lists of following symbols
    symIList = computeList(s, alph)
    # remove last edges from each vertex list, shuffle, then add back
    # the removed edges at end of vertex lists.
    for [x,y] in edgeList:
        symIList[x].remove(y)
    for x in range(len(symIList)):
        random.shuffle(symIList[x])
    for [x,y] in edgeList:
        symIList[x].append(y)
    #construct the eulerian path
    prevSymI = alph.getIndex(s[0])
    L = [alph.getSymbol(prevSymI)]
    for i in range(len(s)-2):
        symI = symIList[prevSymI].pop(0)
        L.append(alph.getSymbol(symI))
        prevSymI = symI
    symI = alph.getIndex(s[-1])
    L.append(alph.getSymbol(symI))
    return "".join(L)
