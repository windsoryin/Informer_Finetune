#import os
#output = os.popen('python main_informer.py --model informer --data hkws_2_5 --data_path hkws_2_5.csv --freq 5t --seq_len 96 --label_len 48 --pred_len 48 --target tp --featrues MS')
#output = os.popen('pwd')

#print(output)
import subprocess
#result = subprocess.run(['python','main_informer.py'], stdout=subprocess.PIPE)
#print(result.stdout.decode('utf-8'))
p = subprocess.Popen(['python test02.py'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in p.stdout.readlines():
    print(line)
retval = p.wait()

