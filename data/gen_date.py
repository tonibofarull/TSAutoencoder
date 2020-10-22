a = 015400.00
print(f"0{round(a,1)}0")

n = 57675
for _ in range(n-1):
    a += 0.2
    if int(a)%100 == 60:
        a = a - 60 + 100
    if (int(a)/100)%100 == 60:
        a = a - 6000 + 10000 
    print(f"0{round(a,1)}0")
    
