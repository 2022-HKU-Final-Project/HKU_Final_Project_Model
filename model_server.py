import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from simpletransformers.classification import ClassificationModel, ClassificationArgs,MultiLabelClassificationModel,MultiLabelClassificationArgs
from sklearn import preprocessing
import pandas as pd
import numpy as np

data = pd.read_csv("./data/train_set_10w_combined.csv")
le = preprocessing.LabelEncoder()
le.fit(np.unique(data['tier2-position'].tolist()))
best_model = ClassificationModel('bert', './best_model/', num_labels=62)

app = FastAPI()

@app.get("/get_info")
async def get_recommendation(content: str):
    if content is None:
        return 'No input'
    else:
        print("content",content)
        # best_model = ClassificationModel('./best_model')
        prediction,_ = best_model.predict(content)
        prediction = le.inverse_transform(prediction)
        print(prediction)
        dic = dict()
        dic['label'] = prediction[0]
        return dic
        


if __name__ == '__main__':
    uvicorn.run(app=app, host='localhost', port=8001)