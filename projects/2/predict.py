#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd

sys.path.append('.')
from model import fields

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#load the model
model = load("2.joblib")

#fields = """doc_id,hotel_name,hotel_url,street,city,state,country,zip,class,price,
#num_reviews,CLEANLINESS,ROOM,SERVICE,LOCATION,VALUE,COMFORT,overall_ratingsource""".replace("\n",'').split(",")

#read and infere
fields.remove('label')
read_opts=dict(
        sep='\t', names=fields, index_col=False, header=None,
        iterator=True, chunksize=100
)

for df in pd.read_csv(sys.stdin, **read_opts):
    pred = model.predict_proba(df)[:, 1]
#     pred = model.predict(df)
    out = zip(df.id, pred)
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))




# import sys, os
# import logging
# from joblib import load
# import pandas as pd

# sys.path.append('.')
# # from model import fields

# #
# # Init the logger
# #
# logging.basicConfig(level=logging.DEBUG)
# logging.info("CURRENT_DIR {}".format(os.getcwd()))
# logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
# logging.info("ARGS {}".format(sys.argv[1:]))

# #load the model
# model = load("2.joblib")

# numeric_features = ["if"+str(i) for i in range(1,14)]
# categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]
# fields = ["id"] + numeric_features + categorical_features
# #read and infere
# read_opts=dict(
#         sep='\t', names=fields, index_col=False, header=None,
#         iterator=True, chunksize=100
# )

# # for df in pd.read_csv(sys.stdin, **read_opts):
# #     pred = model.predict_proba(df)
# #     out = zip(df.id, pred[:, 1])
# #     print("\n".join(["{0}\t{1}".format(*i) for i in out]))
# for line in sys.stdin:
#         nowdata = [x for x in line.strip().split('\t')]
#         pred = model.predict_proba(nowdata)
#         print(pred[1])
#         # out = zip(nowdata[0], pred[1])
#         # # out = pred[:, 1]
#         # print("\n".join(["{0}\t{1}".format(*i) for i in out]))
    