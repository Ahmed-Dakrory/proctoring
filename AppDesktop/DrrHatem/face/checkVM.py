import subprocess
import re


#Check VMWARE
def checkVmWare():
    batcmd='systeminfo /s %computername% | findstr /c:"Model:" /c:"Host Name" /c:"OS Name"'
    result = subprocess.check_output(batcmd, shell=True)
    # print(result)

    if re.search('VirtualBox', str(result), re.IGNORECASE):
        return (True)
    else:
        return (False)

print(checkVmWare())
