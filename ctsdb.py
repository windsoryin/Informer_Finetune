from pandasticsearch import DataFrame
import pandas as pd
import time
# pandas打印所有列，生产环境可以去掉
pd.set_option('display.max_columns', None)
"""
url：
外网地址:http://42.193.247.49:9200   内网地址: http://172.16.20.3:9200

index：
目前已经调试通过2个
1、HWS气象数据：hws@*
2、GNSS数据：gnss@*

username：
账号：gnss

password：
密码：YKY7csX#

verify_ssl：校验SSL证书 False

compat：兼容elasticsearch版本为6.8.2
"""
df = DataFrame.from_es(url='http://42.193.247.49:9200', index="hws@*",
                       username="gnss", password="YKY7csX#", verify_ssl=False,
                       compat=6)
# 固定为doc
df._doc_type = "doc"

# 打印schema 方便查看当前数据格式，生产环境去掉，
# 目前数据格式中除了原本设备数据 还包含 两个时间字段 time设备数据字段  timestamp时序库时间字段（实际查询建议用此字段）。
df.print_schema()

# 查询事例  1、filter：时间过滤、设备过滤； 2、select查询所有字段  3、sort以什么字段进行排序  4、限制返回数量
# 其他用法参考：https://github.com/onesuper/pandasticsearch
data = df.filter((df['timestamp'] > 1681191000) & (df['timestamp'] < 1681191362) & (df['device'] == 'test'))\
    .select(*df.columns)\
    .sort(df["timestamp"].desc)\
    .limit(1000)\
    .to_pandas()
time_data=data.loc[:, 'time']
pa = data.loc[:, 'Pa']
time_np=time_data.to_numpy(dtype=int)
time_local = time.localtime(time_np[0])
print(data)

