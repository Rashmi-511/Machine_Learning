{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c6de8a02",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch \n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ceeca0b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Designing a class for model functionality\n",
    "class Model(nn.Module):\n",
    "    \n",
    "    def __init__(self,in_features=4,h1=8,h2=9,out_features=3):\n",
    "        super().__init__()\n",
    "        self.fc1=nn.Linear(in_features,h1)\n",
    "        self.fc2=nn.Linear(h1,h2)\n",
    "        self.out=nn.Linear(h2,out_features)\n",
    "    \n",
    "    def forward(self,x):\n",
    "        \n",
    "        x=F.relu(self.fc1(x))\n",
    "        x=F.relu(self.fc2(x))\n",
    "        x=self.out(x)\n",
    "        \n",
    "        return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "7b51f3ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "torch.manual_seed(32)\n",
    "model=Model()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "2fdad33d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "15754c21",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_csv('iris.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "476a2573",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \\\n",
       "0                5.1               3.5                1.4               0.2   \n",
       "1                4.9               3.0                1.4               0.2   \n",
       "2                4.7               3.2                1.3               0.2   \n",
       "3                4.6               3.1                1.5               0.2   \n",
       "4                5.0               3.6                1.4               0.2   \n",
       "\n",
       "   target  \n",
       "0     0.0  \n",
       "1     0.0  \n",
       "2     0.0  \n",
       "3     0.0  \n",
       "4     0.0  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "60a1c4ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Defining labels and features\n",
    "X=df.drop('target',axis=1)\n",
    "y=df['target']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "137d4fc5",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Conversion from dataframe and series to column of values\n",
    "X=X.values\n",
    "y=y.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "2249ec89",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Performing train and test split on data\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "1c9ec1ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "eec7fba8",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Conversion of training and testing data to Tensors\n",
    "X_train=torch.FloatTensor(X_train)\n",
    "X_test=torch.FloatTensor(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "661e4ba9",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train=torch.LongTensor(y_train)\n",
    "y_test=torch.LongTensor(y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "2b3cd26f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Defining loss and optimization techniques\n",
    "criterion=nn.CrossEntropyLoss()\n",
    "optimizer=torch.optim.Adam(model.parameters(),lr=0.01)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "5cdf0168",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 0 and loss is 1.1507114171981812\n",
      "Epoch 10 and loss is 0.9377315044403076\n",
      "Epoch 20 and loss is 0.779825747013092\n",
      "Epoch 30 and loss is 0.6099401116371155\n",
      "Epoch 40 and loss is 0.40079933404922485\n",
      "Epoch 50 and loss is 0.25436317920684814\n",
      "Epoch 60 and loss is 0.15053054690361023\n",
      "Epoch 70 and loss is 0.10086946189403534\n",
      "Epoch 80 and loss is 0.08128313720226288\n",
      "Epoch 90 and loss is 0.07231427729129791\n"
     ]
    }
   ],
   "source": [
    "#Data Prediction using the model class considering the training data,implementing Backpropagation and optimization\n",
    "epochs=100\n",
    "losses=[]\n",
    "\n",
    "for i in range(epochs):\n",
    "    y_pred=model.forward(X_train)\n",
    "    loss=criterion(y_pred,y_train)\n",
    "    losses.append(loss)\n",
    "    \n",
    "    if i%10==0:\n",
    "        print(f\"Epoch {i} and loss is {loss}\")\n",
    "        \n",
    "    optimizer.zero_grad()\n",
    "    loss.backward()\n",
    "    optimizer.step()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "7d6e417d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Data validation against the test data by deactivating auto-gradient\n",
    "with torch.no_grad():\n",
    "    y_eval=model.forward(X_test)\n",
    "    loss=criterion(y_eval,y_test)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "c05fded5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor(0.0581)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "5add8c18",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.)1 1\n",
      "2.)1 1\n",
      "3.)0 0\n",
      "4.)1 1\n",
      "5.)2 2\n",
      "6.)2 2\n",
      "7.)0 0\n",
      "8.)0 0\n",
      "9.)2 2\n",
      "10.)2 2\n",
      "11.)2 2\n",
      "12.)0 0\n",
      "13.)2 2\n",
      "14.)1 1\n",
      "15.)2 2\n",
      "16.)1 1\n",
      "17.)2 2\n",
      "18.)0 0\n",
      "19.)1 1\n",
      "20.)2 2\n",
      "21.)0 0\n",
      "22.)0 0\n",
      "23.)2 2\n",
      "24.)0 0\n",
      "25.)2 2\n",
      "26.)2 2\n",
      "27.)1 1\n",
      "28.)1 1\n",
      "29.)2 2\n",
      "30.)2 2\n",
      "We got 30 correct!\n"
     ]
    }
   ],
   "source": [
    "#Displaying predicted values along with actual values and saving the model into a file structure.\n",
    "correct=0\n",
    "\n",
    "with torch.no_grad():\n",
    "    for i,data in enumerate(X_test):\n",
    "        y_val=model.forward(data)\n",
    "        \n",
    "        print(f\"{i+1}.){str(y_val.argmax().item())} {y_test[i]}\")\n",
    "        \n",
    "        if y_val.argmax().item()==y_test[i]:\n",
    "            correct+=1\n",
    "print(f'We got {correct} correct!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "b18eb35e",
   "metadata": {},
   "outputs": [],
   "source": [
    "torch.save(model.state_dict(),'my_iris_model.pt')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "3e5101b1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<All keys matched successfully>"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_model=Model()\n",
    "new_model.load_state_dict(torch.load('my_iris_model.pt'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "3a199acf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Model(\n",
       "  (fc1): Linear(in_features=4, out_features=8, bias=True)\n",
       "  (fc2): Linear(in_features=8, out_features=9, bias=True)\n",
       "  (out): Linear(in_features=9, out_features=3, bias=True)\n",
       ")"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_model.eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1ae74441",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
