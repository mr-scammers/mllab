a=input().split()
d={"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
   "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,
   "thirty":30,"fourty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90,
   }
m={"hundred":100,
   "thousand":1000,
   "million":1000000}
l=[]
n=0
s=0
while n<len(a):
    if a[n] in d:
        l.append(s)
        s=0
        s=s+d[a[n]]
    else:
        s=s*m[a[n]]
    n+=1

l.append(s)


res=sum(l)
print(res)