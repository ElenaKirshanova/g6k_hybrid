def bsearch_component_left(L,x,index):
    # returns entry where the index is the largest number smaller or equal to x in a list of lists that are sorted in index. 
    return bsearch_component_left_iter(L,0,len(L)-1,x,index)

def bsearch_component_left_iter(L,low,high,x,index):
#    print(low, int((high+low)/2), high)
    if high-low<2:
        if L[high][index]<=x:
            return L[high]
        else:
            return L[high-1]
    else:
        mid=int((high+low)/2)
        
        if L[mid][index]>x:
            return bsearch_component_left_iter(L,low,mid-1,x,index)
        elif L[mid][index]<x:
            return bsearch_component_left_iter(L,mid,high,x,index)
        else:
            return bsearch_component_left_iter(L,mid,mid,x,index)

def bsearch_component(L,x,index):
    # returns all entries that contain entry x in index of a list of lists that are sorted in index. 
    return bsearch_component_iter(L,0,len(L)-1,x,index)

def bsearch_component_iter(L,low,high,x,index):
    if high<low:
        return []
    elif high==low:
        if L[low][index]==x:
            true_low=low
            true_high=low
            while true_low>=0 and L[true_low][index]==x:
                true_low-=1
            true_low+=1
            while true_high<len(L) and L[true_high][index]==x:
                true_high+=1
            true_high-=1
            return L[true_low:true_high+1]
        else:
            return []
    else:
        mid=int((high+low)/2)
        if L[mid][index]>x:
            return bsearch_component_iter(L,low,mid-1,x,index)
        elif L[mid][index]<x:
            return bsearch_component_iter(L,mid+1,high,x,index)
        else:
            true_low=mid
            true_high=mid
            while true_low>=0 and L[true_low][index]==x:
                true_low-=1
            true_low+=1
            while true_high<len(L) and L[true_high][index]==x:
                true_high+=1
            true_high-=1
            return L[true_low:true_high+1]