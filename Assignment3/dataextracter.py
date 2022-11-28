import os


x_raw=[]
y_raw=[]

output= open('C_Data.txt','w')
for filename in os.listdir(os.getcwd()+'/Data'):
    input= open('Data/'+filename,'r')
    for line in input:
        l=line.split(';')
        x_raw.append( [int(l[3]),int(l[4])] )
        y_raw.append(int(l[6]))
        t=''
        t=l[3]+','+l[4]+','+l[6]
        output.write( t ) 

for i in range(10):
    print(x_raw[i])

    
