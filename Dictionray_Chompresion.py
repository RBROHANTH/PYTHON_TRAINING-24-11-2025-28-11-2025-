names=["rohanth","santhosh","karthik","vignesh","arun"]
print({x:len(x) for x in names})
dict2=dict(map(lambda x:(x,len(x)),names))
print(dict2)