import sys
import os
import datetime

t = datetime.datetime.now() ## time set to UTC zone
ts = t.strftime("%Y_%m_%d_ %H_%M_%S") # Date format


if len(sys.argv) != 2: # if CLi does not equal to 2 commands print
	print ("usage:toCSV.py logname.ext")
	sys.exit(1)

logSys = sys.argv[1] 
newLogSys = sys.argv[1] + "_.csv"

log = open(logSys,"r")
nL = file(newLogSys  ,"w") 


# Read from file log and write to nLog file
for lineI in log.readlines():
    rec=lineI.rstrip()
    if rec.startswith("#"):
        lineI=rec.replace(':',',').strip() 
        nL.write(lineI + "\n")
    else:
    	rec=rec.split(' ')
    	rec= [w for w in rec if w!='']
    	print '\t'.join(rec) 
        #lineO=rec.replace(' ','\t').strip() #
        nL.write('\t'.join(rec)  + "\n") 

## closes both open files; End script
nL.close()
log.close()
