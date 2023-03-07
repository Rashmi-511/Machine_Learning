{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4e39f178",
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a40946df",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "14f4845f",
   "metadata": {},
   "outputs": [],
   "source": [
    "messages=[line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5b6b4940",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5574\n"
     ]
    }
   ],
   "source": [
    "print(len(messages))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d61a9bd1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'ham\\tWhat you thinked about me. First time you saw me in class.'"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages[50]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "56403093",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...\n",
      "\n",
      "\n",
      "1 ham\tOk lar... Joking wif u oni...\n",
      "\n",
      "\n",
      "2 spam\tFree entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's\n",
      "\n",
      "\n",
      "3 ham\tU dun say so early hor... U c already then say...\n",
      "\n",
      "\n",
      "4 ham\tNah I don't think he goes to usf, he lives around here though\n",
      "\n",
      "\n",
      "5 spam\tFreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, Â£1.50 to rcv\n",
      "\n",
      "\n",
      "6 ham\tEven my brother is not like to speak with me. They treat me like aids patent.\n",
      "\n",
      "\n",
      "7 ham\tAs per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune\n",
      "\n",
      "\n",
      "8 spam\tWINNER!! As a valued network customer you have been selected to receivea Â£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.\n",
      "\n",
      "\n",
      "9 spam\tHad your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "for mess_no,message in enumerate(messages[:10]):\n",
    "    print(mess_no,message)\n",
    "    print('\\n')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "3e9181f8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'ham\\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...'"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "0a17d1d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "messages=pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\\t',names=['label','message'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "38f19ffe",
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
       "      <th>label</th>\n",
       "      <th>message</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  label                                            message\n",
       "0   ham  Go until jurong point, crazy.. Available only ...\n",
       "1   ham                      Ok lar... Joking wif u oni...\n",
       "2  spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3   ham  U dun say so early hor... U c already then say...\n",
       "4   ham  Nah I don't think he goes to usf, he lives aro..."
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "b305cddb",
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
       "      <th>label</th>\n",
       "      <th>message</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>5572</td>\n",
       "      <td>5572</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>2</td>\n",
       "      <td>5169</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>ham</td>\n",
       "      <td>Sorry, I'll call later</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>4825</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       label                 message\n",
       "count   5572                    5572\n",
       "unique     2                    5169\n",
       "top      ham  Sorry, I'll call later\n",
       "freq    4825                      30"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "045a695b",
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
       "    .dataframe thead tr th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr:last-of-type th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th colspan=\"4\" halign=\"left\">message</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>unique</th>\n",
       "      <th>top</th>\n",
       "      <th>freq</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>label</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>ham</th>\n",
       "      <td>4825</td>\n",
       "      <td>4516</td>\n",
       "      <td>Sorry, I'll call later</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>spam</th>\n",
       "      <td>747</td>\n",
       "      <td>653</td>\n",
       "      <td>Please call our customer service representativ...</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      message                                                               \n",
       "        count unique                                                top freq\n",
       "label                                                                       \n",
       "ham      4825   4516                             Sorry, I'll call later   30\n",
       "spam      747    653  Please call our customer service representativ...    4"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages.groupby('label').describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "1b2d2797",
   "metadata": {},
   "outputs": [],
   "source": [
    "messages['length']=messages['message'].apply(len)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "2fa03a59",
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
       "      <th>label</th>\n",
       "      <th>message</th>\n",
       "      <th>length</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "      <td>111</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "      <td>29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "      <td>155</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "      <td>49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "      <td>61</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  label                                            message  length\n",
       "0   ham  Go until jurong point, crazy.. Available only ...     111\n",
       "1   ham                      Ok lar... Joking wif u oni...      29\n",
       "2  spam  Free entry in 2 a wkly comp to win FA Cup fina...     155\n",
       "3   ham  U dun say so early hor... U c already then say...      49\n",
       "4   ham  Nah I don't think he goes to usf, he lives aro...      61"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "b5ae56c0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:ylabel='Frequency'>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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+vTpam5ujlSfAAAAYTmhsLNz507NmTNH55xzjiorK1VWVqaPP/5Yr7/+ug4cOKDrrrsuUn0CAACEJT6cH6qsrFRVVZU++OADXXPNNXrmmWd0zTXX6IwzvslOOTk5evLJJ3XBBRdEtFkAAIDjFVbYWbt2rWbOnKk777xTXq+335qhQ4dq3bp1J9QcAADAiQor7Hz00UdHrUlMTNSMGTPCuTwAAEDEhLVnp6qqSs8//3yf8eeff14bNmw44aYAAAAiJayVneXLl+uJJ57oMz548GD98z//Mys6A2T44hf7jH26fGIUOgEAIHaFtbKzd+9e5eTk9BkfNmyY9u3bd8JNAQAAREpYYWfw4MF67733+oy/++67ysjIOOGmAAAAIiWssHPzzTfr3nvv1RtvvKGenh719PTo9ddf1/z583XzzTdHukcAAICwhbVn5+GHH9bevXtVVFSk+PhvLtHb26vp06ervLw8og0CAACciLDCTmJiop577jn967/+q959910lJSUpPz9fw4YNi3R/AAAAJySssPOt888/X+eff36kegEAAIi4sMJOT0+P1q9fr9dee00tLS3q7e0NOf/6669HpDkAAIATFVbYmT9/vtavX6+JEycqLy9PLpcr0n0BAABERFhhp7q6Wr/97W91zTXXRLofAACAiArro+eJiYk677zzIt0LAABAxIUVdhYuXKhf/epXMsZEuh8AAICICuttrLq6Or3xxht6+eWXddFFFykhISHk/ObNmyPSHAAAwIkKK+ycddZZuuGGGyLdCwAAQMSFFXaqqqoi3QcAAMBJEdaeHUn6+uuvtW3bNj355JNqa2uTJB08eFDt7e1hXa+iokIul0ulpaXOmDFGy5Ytk8/nU1JSksaOHas9e/aE/FxnZ6fmzZunzMxMpaSkaPLkyWpqagr3tgAAgGXCCjt79+5Vfn6+rrvuOs2dO1d/+9vfJEkrVqxQWVnZcV9vx44deuqpp3TxxReHjK9YsUKVlZVas2aNduzYIa/Xq3HjxjnhSpJKS0u1ZcsWVVdXq66uTu3t7Zo0aZJ6enrCuTUAAGCZsMLO/PnzNXLkSLW2tiopKckZv+GGG/Taa68d17Xa29t166236umnn9agQYOccWOMVq9eraVLl2rKlCnKy8vThg0b9OWXX2rTpk2SpEAgoHXr1unRRx9VcXGxRowYoY0bN2r37t3atm1bOLcGAAAsE1bYqaur0y9+8QslJiaGjA8bNkwHDhw4rmvNnTtXEydOVHFxcch4Y2Oj/H6/SkpKnDG3260xY8Zo+/btkqT6+np1d3eH1Ph8PuXl5Tk1/ens7FQwGAw5AACAncLaoNzb29vv20RNTU1KTU095utUV1ervr5eO3fu7HPO7/dLkrKyskLGs7KytHfvXqcmMTExZEXo25pvf74/FRUVevDBB4+5TwAAcOoKa2Vn3LhxWr16tfPa5XKpvb1dDzzwwDE/QmL//v2aP3++fv3rX+vMM888Yt3hz90yxhz1WVxHq1myZIkCgYBz7N+//5h6BgAAp56wws6qVatUW1urCy+8UF999ZWmTZum4cOH68CBA3rkkUeO6Rr19fVqaWlRQUGB4uPjFR8fr9raWj322GOKj493VnQOX6FpaWlxznm9XnV1dam1tfWINf1xu91KS0sLOQAAgJ3CCjs+n0+7du1SWVmZZs+erREjRmj58uV65513NHjw4GO6RlFRkXbv3q1du3Y5x8iRI3Xrrbdq165dOvfcc+X1elVTU+P8TFdXl2pra1VYWChJKigoUEJCQkhNc3OzGhoanBoAAHB6C2vPjiQlJSVp5syZmjlzZlg/n5qaqry8vJCxlJQUZWRkOOOlpaUqLy9Xbm6ucnNzVV5eruTkZE2bNk2S5PF4NGvWLC1cuFAZGRlKT09XWVmZ8vPz+2x4BgAAp6ewws4zzzzzneenT58eVjOHW7RokTo6OjRnzhy1trZq1KhR2rp1a8gm6FWrVik+Pl5Tp05VR0eHioqKtH79esXFxUWkBwAAcGpzmTAeXX74p5+6u7v15ZdfKjExUcnJyfriiy8i1uBACAaD8ng8CgQCUdm/M3zxixG71qfLJ0bsWgAAxLJj/fsd1p6d1tbWkKO9vV0ffPCBRo8erd/85jdhNw0AABBpYT8b63C5ublavny55s+fH6lLAgAAnLCIhR1JiouL08GDByN5SQAAgBMS1gblF154IeS1MUbNzc1as2aNrrjiiog0BgAAEAlhhZ3rr78+5LXL5dLZZ5+tq6++Wo8++mgk+gIAAIiIsJ+NBQAAcCqI6J4dAACAWBPWys6CBQuOubaysjKcXwEAABARYYWdd955R2+//ba+/vpr/fCHP5Qkffjhh4qLi9Oll17q1B3t6eQAAAAnW1hh59prr1Vqaqo2bNjgfJtya2ur7rzzTv30pz/VwoULI9okAABAuMLas/Poo4+qoqIi5LERgwYN0sMPP8ynsQAAQEwJK+wEg0H99a9/7TPe0tKitra2E24KAAAgUsIKOzfccIPuvPNO/e53v1NTU5Oampr0u9/9TrNmzdKUKVMi3SMAAEDYwtqz88QTT6isrEy33Xaburu7v7lQfLxmzZqllStXRrRB2O/wp77z5HYAQCSFFXaSk5P1+OOPa+XKlfr4449ljNF5552nlJSUSPcHAABwQk7oSwWbm5vV3Nys888/XykpKTLGRKovAACAiAgr7Hz++ecqKirS+eefr2uuuUbNzc2SpLvuuouPnQMAgJgSVti57777lJCQoH379ik5OdkZv+mmm/TKK69ErDkAAIATFdaena1bt+rVV1/VkCFDQsZzc3O1d+/eiDQGAAAQCWGt7Bw6dChkRedbn332mdxu9wk3BQAAEClhhZ0rr7xSzzzzjPPa5XKpt7dXK1eu1FVXXRWx5gAAAE5UWG9jrVy5UmPHjtXOnTvV1dWlRYsWac+ePfriiy/05z//OdI9AgAAhC2slZ0LL7xQ7733nn7yk59o3LhxOnTokKZMmaJ33nlHP/jBDyLdIwAAQNiOe2Wnu7tbJSUlevLJJ/Xggw+ejJ4AAAAi5rhXdhISEtTQ0CCXy3Uy+gEAAIiosN7Gmj59utatWxfpXgAAACIurA3KXV1d+vd//3fV1NRo5MiRfZ6JVVlZGZHmAAAATtRxhZ1PPvlEw4cPV0NDgy699FJJ0ocffhhSw9tbAAAglhxX2MnNzVVzc7PeeOMNSd88HuKxxx5TVlbWSWkOAADgRB3Xnp3Dn2r+8ssv69ChQxFtCAAAIJLC2qD8rcPDDwAAQKw5rrDjcrn67Mlhjw4AAIhlx7VnxxijO+64w3nY51dffaW77767z6exNm/eHLkOAQAATsBxhZ0ZM2aEvL7tttsi2gwAAECkHVfYqaqqOll9AAAAnBQntEEZAAAg1kU17Kxdu1YXX3yx0tLSlJaWpssvv1wvv/yyc94Yo2XLlsnn8ykpKUljx47Vnj17Qq7R2dmpefPmKTMzUykpKZo8ebKampoG+lYAAECMimrYGTJkiJYvX66dO3dq586duvrqq3Xdddc5gWbFihWqrKzUmjVrtGPHDnm9Xo0bN05tbW3ONUpLS7VlyxZVV1errq5O7e3tmjRpknp6eqJ1W1E1fPGLIQcAAKc7l4mxL8tJT0/XypUrNXPmTPl8PpWWlur++++X9M0qTlZWlh555BHNnj1bgUBAZ599tp599lnddNNNkqSDBw8qOztbL730ksaPH39MvzMYDMrj8SgQCCgtLe2k3duRDHQo+XT5xAH9fUdz+P3HWn8AgNh0rH+/Y2bPTk9Pj6qrq3Xo0CFdfvnlamxslN/vV0lJiVPjdrs1ZswYbd++XZJUX1+v7u7ukBqfz6e8vDynpj+dnZ0KBoMhBwAAsFNYTz2PpN27d+vyyy/XV199pe9973vasmWLLrzwQiesHP7craysLO3du1eS5Pf7lZiYqEGDBvWp8fv9R/ydFRUVevDBByN8J4iU/la6WO0BAIQr6is7P/zhD7Vr1y69+eab+tnPfqYZM2bo/fffd84f/g3Nxpijfmvz0WqWLFmiQCDgHPv37z+xmwAAADEr6mEnMTFR5513nkaOHKmKigpdcskl+tWvfiWv1ytJfVZoWlpanNUer9errq4utba2HrGmP2632/kE2LcHAACwU9TDzuGMMers7FROTo68Xq9qamqcc11dXaqtrVVhYaEkqaCgQAkJCSE1zc3NamhocGoAAMDpLap7dn7+859rwoQJys7OVltbm6qrq/XHP/5Rr7zyilwul0pLS1VeXq7c3Fzl5uaqvLxcycnJmjZtmiTJ4/Fo1qxZWrhwoTIyMpSenq6ysjLl5+eruLg4mrcGAABiRFTDzl//+lfdfvvtam5ulsfj0cUXX6xXXnlF48aNkyQtWrRIHR0dmjNnjlpbWzVq1Cht3bpVqampzjVWrVql+Ph4TZ06VR0dHSoqKtL69esVFxcXrdsCAAAxJOa+Zyca+J6d6DqW+4+1ngEA0XfKfc8OAADAyUDYAQAAViPsAAAAqxF2AACA1Qg7AADAaoQdAABgNcIOAACwGmEHAABYjbADAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBqhB0AAGA1wg4AALAaYQcAAFiNsAMAAKxG2AEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWI2wAwAArBYf7QYw8IYvfjHk9afLJ0apEwAATj5WdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWI2wAwAArEbYAQAAVuN7djCgDv+OHwAATjbCDvoNIHzRIADAFryNBQAArBbVsFNRUaHLLrtMqampGjx4sK6//np98MEHITXGGC1btkw+n09JSUkaO3as9uzZE1LT2dmpefPmKTMzUykpKZo8ebKampoG8lYAAECMimrYqa2t1dy5c/Xmm2+qpqZGX3/9tUpKSnTo0CGnZsWKFaqsrNSaNWu0Y8cOeb1ejRs3Tm1tbU5NaWmptmzZourqatXV1am9vV2TJk1ST09PNG4LAADEkKju2XnllVdCXldVVWnw4MGqr6/XlVdeKWOMVq9eraVLl2rKlCmSpA0bNigrK0ubNm3S7NmzFQgEtG7dOj377LMqLi6WJG3cuFHZ2dnatm2bxo8fP+D3BQAAYkdM7dkJBAKSpPT0dElSY2Oj/H6/SkpKnBq3260xY8Zo+/btkqT6+np1d3eH1Ph8PuXl5Tk1h+vs7FQwGAw5AACAnWIm7BhjtGDBAo0ePVp5eXmSJL/fL0nKysoKqc3KynLO+f1+JSYmatCgQUesOVxFRYU8Ho9zZGdnR/p2AABAjIiZsHPPPffovffe029+85s+51wuV8hrY0yfscN9V82SJUsUCAScY//+/eE3DgAAYlpMhJ158+bphRde0BtvvKEhQ4Y4416vV5L6rNC0tLQ4qz1er1ddXV1qbW09Ys3h3G630tLSQg4AAGCnqIYdY4zuuecebd68Wa+//rpycnJCzufk5Mjr9aqmpsYZ6+rqUm1trQoLCyVJBQUFSkhICKlpbm5WQ0ODUwMAAE5fUf001ty5c7Vp0yb953/+p1JTU50VHI/Ho6SkJLlcLpWWlqq8vFy5ubnKzc1VeXm5kpOTNW3aNKd21qxZWrhwoTIyMpSenq6ysjLl5+c7n84CAACnr6iGnbVr10qSxo4dGzJeVVWlO+64Q5K0aNEidXR0aM6cOWptbdWoUaO0detWpaamOvWrVq1SfHy8pk6dqo6ODhUVFWn9+vWKi4sbqFvBEfAsLABAtLmMMSbaTURbMBiUx+NRIBCIyv6dWAwEkXo2VqTujWd1AQAOd6x/v2NigzIAAMDJQtgBAABWI+wAAACrEXYAAIDVCDsAAMBqhB0AAGA1wg4AALAaYQcAAFiNsAMAAKxG2AEAAFYj7AAAAKsRdgAAgNWi+tRznNoOf8gnD+sEAMQiwg4iJhaf3g4AAG9jAQAAqxF2AACA1XgbC/1iPw4AwBaEHRwT9uMAAE5VvI0FAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBqhB0AAGA1wg4AALAaYQcAAFiNsAMAAKxG2AEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWI2wAwAArEbYAQAAVotq2Pnv//5vXXvttfL5fHK5XPr9738fct4Yo2XLlsnn8ykpKUljx47Vnj17Qmo6Ozs1b948ZWZmKiUlRZMnT1ZTU9MA3gUAAIhlUQ07hw4d0iWXXKI1a9b0e37FihWqrKzUmjVrtGPHDnm9Xo0bN05tbW1OTWlpqbZs2aLq6mrV1dWpvb1dkyZNUk9Pz0DdBgAAiGHx0fzlEyZM0IQJE/o9Z4zR6tWrtXTpUk2ZMkWStGHDBmVlZWnTpk2aPXu2AoGA1q1bp2effVbFxcWSpI0bNyo7O1vbtm3T+PHjB+xeAABAbIrZPTuNjY3y+/0qKSlxxtxut8aMGaPt27dLkurr69Xd3R1S4/P5lJeX59T0p7OzU8FgMOQAAAB2itmw4/f7JUlZWVkh41lZWc45v9+vxMREDRo06Ig1/amoqJDH43GO7OzsCHcPAABiRVTfxjoWLpcr5LUxps/Y4Y5Ws2TJEi1YsMB5HQwGBzTwDF/84oD9LgAATncxu7Lj9Xolqc8KTUtLi7Pa4/V61dXVpdbW1iPW9MftdistLS3kAAAAdorZsJOTkyOv16uamhpnrKurS7W1tSosLJQkFRQUKCEhIaSmublZDQ0NTg0AADi9RfVtrPb2dv3lL39xXjc2NmrXrl1KT0/X0KFDVVpaqvLycuXm5io3N1fl5eVKTk7WtGnTJEkej0ezZs3SwoULlZGRofT0dJWVlSk/P9/5dBYAADi9RTXs7Ny5U1dddZXz+tt9NDNmzND69eu1aNEidXR0aM6cOWptbdWoUaO0detWpaamOj+zatUqxcfHa+rUqero6FBRUZHWr1+vuLi4Ab8fAAAQe1zGGBPtJqItGAzK4/EoEAgMyP4dNigfv0+XT4x2CwCAGHOsf79jds8OAABAJBB2AACA1Qg7AADAaoQdAABgNcIOAACwGmEHAABYjbADAACsRtgBAABWI+wAAACrRfVxEacDvi0ZAIDoYmUHAABYjbADAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBqhB0AAGA1wg4AALAaYQcAAFiNsAMAAKxG2AEAAFbj2Vg4JRz+jLFPl0+MUicAgFMNKzsAAMBqhB0AAGA1wg4AALAaYQcAAFiNDco4JR2+YVli0zIAoH+s7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWI2wAwAArMb37MAaPCwUANAfwg6s1d8XDx6OQAQA9rMm7Dz++ONauXKlmpubddFFF2n16tX66U9/Gu22YIGT+W3NrEYBwMlnRdh57rnnVFpaqscff1xXXHGFnnzySU2YMEHvv/++hg4dGu32cIo5lhWhgQwp4fwuHqcBAP/HirBTWVmpWbNm6a677pIkrV69Wq+++qrWrl2rioqKKHeH00GkwsWxBK1w2bqKZOt9AYicUz7sdHV1qb6+XosXLw4ZLykp0fbt2/v9mc7OTnV2djqvA4GAJCkYDEa8v97OLyN+TUTO0PueP+Wu3d//p3kPvBqR6xyLY/ldDQ+OD+va4Tj839ix/PcZyP6irb/5Op3uH3b79t+7MeY76075sPPZZ5+pp6dHWVlZIeNZWVny+/39/kxFRYUefPDBPuPZ2dknpUcgkjyrY+s6A33tSPzuaPYXC073+4d92tra5PF4jnj+lA8733K5XCGvjTF9xr61ZMkSLViwwHnd29urL774QhkZGUf8meMVDAaVnZ2t/fv3Ky0tLSLXRPiYj9jBXMQO5iK2MB/HzxijtrY2+Xy+76w75cNOZmam4uLi+qzitLS09Fnt+Zbb7Zbb7Q4ZO+uss05Kf2lpafxPG0OYj9jBXMQO5iK2MB/H57tWdL51yn+DcmJiogoKClRTUxMyXlNTo8LCwih1BQAAYsUpv7IjSQsWLNDtt9+ukSNH6vLLL9dTTz2lffv26e677452awAAIMqsCDs33XSTPv/8cz300ENqbm5WXl6eXnrpJQ0bNixqPbndbj3wwAN93i5DdDAfsYO5iB3MRWxhPk4elzna57UAAABOYaf8nh0AAIDvQtgBAABWI+wAAACrEXYAAIDVCDsnyeOPP66cnBydeeaZKigo0J/+9Kdot2SViooKXXbZZUpNTdXgwYN1/fXX64MPPgipMcZo2bJl8vl8SkpK0tixY7Vnz56Qms7OTs2bN0+ZmZlKSUnR5MmT1dTUNJC3Yp2Kigq5XC6VlpY6Y8zFwDpw4IBuu+02ZWRkKDk5WT/+8Y9VX1/vnGc+BsbXX3+tX/ziF8rJyVFSUpLOPfdcPfTQQ+rt7XVqmIsBYhBx1dXVJiEhwTz99NPm/fffN/PnzzcpKSlm79690W7NGuPHjzdVVVWmoaHB7Nq1y0ycONEMHTrUtLe3OzXLly83qamp5j/+4z/M7t27zU033WTOOeccEwwGnZq7777bfP/73zc1NTXm7bffNldddZW55JJLzNdffx2N2zrlvfXWW2b48OHm4osvNvPnz3fGmYuB88UXX5hhw4aZO+64w/zP//yPaWxsNNu2bTN/+ctfnBrmY2A8/PDDJiMjw/zhD38wjY2N5vnnnzff+973zOrVq50a5mJgEHZOgp/85Cfm7rvvDhm74IILzOLFi6PUkf1aWlqMJFNbW2uMMaa3t9d4vV6zfPlyp+arr74yHo/HPPHEE8YYY/7+97+bhIQEU11d7dQcOHDAnHHGGeaVV14Z2BuwQFtbm8nNzTU1NTVmzJgxTthhLgbW/fffb0aPHn3E88zHwJk4caKZOXNmyNiUKVPMbbfdZoxhLgYSb2NFWFdXl+rr61VSUhIyXlJSou3bt0epK/sFAgFJUnp6uiSpsbFRfr8/ZB7cbrfGjBnjzEN9fb26u7tDanw+n/Ly8pirMMydO1cTJ05UcXFxyDhzMbBeeOEFjRw5UjfeeKMGDx6sESNG6Omnn3bOMx8DZ/To0Xrttdf04YcfSpLeffdd1dXV6ZprrpHEXAwkK75BOZZ89tln6unp6fMQ0qysrD4PK0VkGGO0YMECjR49Wnl5eZLk/Lfubx727t3r1CQmJmrQoEF9apir41NdXa36+nrt3LmzzznmYmB98sknWrt2rRYsWKCf//zneuutt3TvvffK7XZr+vTpzMcAuv/++xUIBHTBBRcoLi5OPT09+uUvf6lbbrlFEv82BhJh5yRxuVwhr40xfcYQGffcc4/ee+891dXV9TkXzjwwV8dn//79mj9/vrZu3aozzzzziHXMxcDo7e3VyJEjVV5eLkkaMWKE9uzZo7Vr12r69OlOHfNx8j333HPauHGjNm3apIsuuki7du1SaWmpfD6fZsyY4dQxFycfb2NFWGZmpuLi4vok7paWlj7pHSdu3rx5euGFF/TGG29oyJAhzrjX65Wk75wHr9errq4utba2HrEGR1dfX6+WlhYVFBQoPj5e8fHxqq2t1WOPPab4+HjnvyVzMTDOOeccXXjhhSFjP/rRj7Rv3z5J/NsYSP/yL/+ixYsX6+abb1Z+fr5uv/123XfffaqoqJDEXAwkwk6EJSYmqqCgQDU1NSHjNTU1KiwsjFJX9jHG6J577tHmzZv1+uuvKycnJ+R8Tk6OvF5vyDx0dXWptrbWmYeCggIlJCSE1DQ3N6uhoYG5Og5FRUXavXu3du3a5RwjR47Urbfeql27duncc89lLgbQFVdc0edrGD788EPnwcj82xg4X375pc44I/TPbFxcnPPRc+ZiAEVpY7TVvv3o+bp168z7779vSktLTUpKivn000+j3Zo1fvaznxmPx2P++Mc/mubmZuf48ssvnZrly5cbj8djNm/ebHbv3m1uueWWfj/SOWTIELNt2zbz9ttvm6uvvpqPdEbA//9pLGOYi4H01ltvmfj4ePPLX/7SfPTRR+bXv/61SU5ONhs3bnRqmI+BMWPGDPP973/f+ej55s2bTWZmplm0aJFTw1wMDMLOSfJv//ZvZtiwYSYxMdFceumlzkeiERmS+j2qqqqcmt7eXvPAAw8Yr9dr3G63ufLKK83u3btDrtPR0WHuuecek56ebpKSksykSZPMvn37Bvhu7HN42GEuBtZ//dd/mby8PON2u80FF1xgnnrqqZDzzMfACAaDZv78+Wbo0KHmzDPPNOeee65ZunSp6ezsdGqYi4HhMsaYaK4sAQAAnEzs2QEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWI2wAwAArEbYAQAAViPsAAAAqxF2AACA1Qg7AADAav8PYpcIfS2dcjwAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "messages['length'].plot.hist(bins=100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "96c86b72",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    5572.000000\n",
       "mean       80.489950\n",
       "std        59.942907\n",
       "min         2.000000\n",
       "25%        36.000000\n",
       "50%        62.000000\n",
       "75%       122.000000\n",
       "max       910.000000\n",
       "Name: length, dtype: float64"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages['length'].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "75eecfe0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"For me the love should start with attraction.i should feel that I need her every time around me.she should be the first thing which comes in my thoughts.I would start the day and end it with her.she should be there every time I dream.love will be then when my every breath has her name.my life should happen around her.my life will be named to her.I would cry for her.will give all my happiness and take all her sorrows.I will be ready to fight with anyone for her.I will be in love when I will be doing the craziest things for her.love will be when I don't have to proove anyone that my girl is the most beautiful lady on the whole planet.I will always be singing praises for her.love will be when I start up making chicken curry and end up makiing sambar.life will be the most beautiful then.will get every morning and thank god for the day because she is with me.I would like to say a lot..will tell later..\""
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages[messages['length']==910]['message'].iloc[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "c717e40a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([<AxesSubplot:title={'center':'ham'}>,\n",
       "       <AxesSubplot:title={'center':'spam'}>], dtype=object)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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AAAAAIsgBAAAAAAAwCIIAQAAAAAAsIhufYvAYDhw4IBWrFihyspKud1u2Ww2ORwOpaWl6cYbb1RCQkKwSwQAAAAA4JRwi8Cv2Lp1q8aPH6+EhARlZmbK4XDIMAw1NDSovLxcdXV1eu655zRq1Khgl9pjGIahLVu2dAhVRo0apTFjxshmswW7xB6DsfQtxtO3GE8AABBMzEW+OUKArzj33HN1wQUXaNmyZSfcf9ttt2nr1q2qqqoKcGU90wcffKCJEydq586dSk5O9gpVdu3apeHDh2vDhg067bTTgl1qt8dY+hbj6VuMJwBfOnLkiMrKyk44kb/88ssVHR0d7BJDBh+aAoNx9j/mIl1DCPAVkZGR2r59u37wgx+ccP8///lPnX322Wpubg5wZT3TxRdfrMOHD2vt2rUaNGiQ1776+npdeeWViomJ0VNPPRWcAnsQxtK3GE/fYjwB+Mpbb72ljIwMHT16VOnp6V4T+YqKCkVHR2vz5s364Q9/GOxSezw+NAUG4xwYzEW6hhDgK8444wzdeeeduvbaa0+4/7HHHtNvfvMbvffeewGurGf6zne+o1deeUXDhw8/4f4333xTF154oQ4fPhzgynoextK3GE/fYjwB+MpFF10kp9Op0tJSRUREeO1rbW3VNddco/r6ev3jH/8IUoWhgw9NgcE4BwZzka7hwoBfMXfuXN14442qqalRRkaGHA6HbDab3G63ysvL9cgjj6ioqCjYZfYYkZGR+uSTTzrd39jYqMjIyABW1HMxlr7FePoW4wnAV15//XVVV1d3CAAkKSIiQrfffrvOO++8IFQWel544QW98sorHT6YStKgQYN0//3368ILLwxCZaGFcQ4M5iJdwy0Cv2LmzJl6/PHHVV1drUsvvVRpaWk6//zzdemll6q6ulqPP/64brzxxmCX2WNcdtllmjZtmp588kl5PB6z3ePx6Mknn9S1116rrKysIFbYczCWvsV4+hbjCcBX+vfvr3feeafT/e+++6769+8fwIpCFx+aAoNxDgzmIl1k4IRaW1uNDz/80Pjwww+N1tbWYJfTI7W0tBg33nijERERYfTq1cvo06eP0adPH6NXr15GRESEcdNNNxktLS3BLrNHYCx9i/H0LcYTgK/cfffdRmxsrHHfffcZ27dvN+rr6w23221s377duO+++4z+/fsbCxcuDHaZIeGWW24xEhISjD//+c/Gp59+arZ/+umnxp///Gdj8ODBxq233hrECkMD4xwYzEW6hmsCwO+amppUXV2tjz76SJLkdDqVmpqqvn37Brmynoex9C3G07cYTwC+sGTJEj344IPmVdSlL66u7nQ6lZ2drZycnCBXGBpaW1v1q1/9So8++qg+//xz8xSM1tZWhYeHa/r06SoqKjrhqRn45hjnwGIu8s0QAgAAAKDbqa2tldvtlvTFRD4xMTHIFYUmPjQFRlNTk2pqarz+TTPOCBZCAPgV9/r1HcbStxhP32I8AfhDY2OjSktL9c4778jlcunqq69WQkJCsMsC0A0xF/nmCAHgN9zr13cYS99iPH2L8QTgKy6XSzt37lR8fLxqa2s1atQoGYahlJQUvf322zp06JBee+01DR06NNilhgQ+NAVeW1ubNm7cqHfeeUeDBg3S//7v/zLOPsBcpGsIAeA33OvXdxhL32I8fYvxBOArvXr1ktvt1sCBA3X55ZfL7XZr48aNioqKUktLiy699FL16dNHf/7zn4Ndao/Hh6bASEtL07PPPqt+/frp448/1k9+8hPt3btXp59+uurq6jRw4EBVVlbqtNNOC3apPRpzka4hBIDfREVFqbq6utM3j127dum8887T0aNHA1xZz8NY+hbj6VuMJwBf+WoIcMYZZ+iRRx7RT37yE3P/66+/rksvvVR1dXVBrDI08KEpML76b/qGG25QVVWVnnvuOTmdTh08eFCTJ0/W0KFDtXr16mCX2qMxF+maXsEuAKGLe/36DmPpW4ynbzGeAHzp+B0BWlpa5HA4vPY5HA59/PHHwSgr5Lz++uu68847T3hV+oiICN1+++16/fXXg1BZ6KqoqNA999wjp9MpSYqPj9fixYv197//PciV9XzMRbomPNgFIHRdf/31mjZtmu644w5lZGTI4XDIZrPJ7XarvLxc+fn5ys7ODnaZPQJj6VuMp28xngB8acyYMQoPD1dTU5P27t2rM88809y3f/9+DRgwIIjVhY7jH5o6++aUD02+czzY+vTTTzvc5SIxMVH19fXBKCukMBfpGkIA+E1eXp4iIyNVWFionJycDvf6nT9/Pvf6/YYYS99iPH2L8QTgK3fffbfXdlRUlNf2008/rQsvvDCQJYUsPjQFzjXXXCO73a62tjbt27fPK3ipr69Xv379gldciGAu0jVcEwABwb1+feerY+lwOHTGGWcEuaKejX+bvsV4AkDPsWTJEj344IPmnQGkLz80ZWdn86HJB6699lqv7Z/97Gf6xS9+YW7/+te/1s6dO7Vp06ZAlxaymIv8Z4QAQA8WERGhHTt2aNiwYcEuBQAA9FB8aAqeI0eOKCwsTH369Al2KbAQTgeAXzU3N6umpkZxcXEdzjn77LPP9Kc//UlXX311kKrrOWbPnn3C9vb2dt17772Kj4+XJBUWFgayrB7rzTffVL9+/cxJztq1a7VixQrt379fp59+um655RZddtllQa6yZ1m+fLmqq6s1YcIETZ06VWvWrFFBQYGOHTumKVOmaNGiRQoP5y0HALqjxMTEDh/86+rqdPfdd+vRRx8NUlXW8MknnzDOPsLnjm+OlQDwm7179yozM1P79++XzWbThRdeqD/+8Y8aNGiQJOmjjz6Sy+VSe3t7kCvt/nr16qXhw4d3OGesoqJCI0aMUHR0tGw2G1eX/YbOOeccPfDAA7rooov0yCOP6NZbb9X111+vYcOGac+ePXrkkUf04IMP6rrrrgt2qT3Cb37zG913333KzMzUK6+8ouzsbN1333267bbb1KtXLy1btkw33XSTFi5cGOxSAQDf0I4dO3TOOecwT/Mzxtk3+NzRNXwtA7+ZN2+eUlJSVF1drU8//VSzZ8/WqFGj9OKLL2rw4MHBLq9HWbx4sVatWqUHHnjA637JvXv3VklJSadX9sWJ7dmzR//1X/8lSXr44YdVVFSkG264wdx/7rnnavHixYQA31BJSYlKSko0ZcoU7dixQ6mpqSotLdUVV1whSRo6dKhycnIIAQCgG9mwYcNJ97/33nsBqiS0Mc6BweeOrmElAPzG4XBoy5YtSklJMdtuvvlmPfPMM/rHP/6h6OhoErkuqKqq0pVXXqlJkyapoKBAvXv3Vu/evbVjxw5CgC4aMGCAnn/+eaWmpsrhcGjz5s0aPny4uf9f//qXUlJSdPTo0SBW2XNERUXpn//8p/kmGxERoTfffNO8rdfxKyEfOXIkmGUCAL6iV69estlsOtlHAZvNxjztW2KcA4PPHV3TK9gFIHQ1Nzd3OAf4oYce0uTJk5Wenq69e/cGqbKe6dxzz1VNTY0+/vhjpaamaufOneaVfNE148eP14oVKyRJ6enpevLJJ732/+lPf9J///d/B6O0HsnpdOqtt96SJL3zzjtqb283tyVp9+7dGjhwYLDKAwCcwKBBg/SXv/xFx44dO+Fj27ZtwS4xJDDOgcHnjq7hdAD4zdChQ1VdXd3hyvXLly+XYRiaPHlykCrrub7zne+otLRU69atU0ZGBmnmKVqyZIlGjRql9PR0jRgxQg888IBefPFF85oAr732mtavXx/sMnuMrKwsXX311br44ov1wgsvaN68eZo7d64OHjwom82mxYsX69JLLw12mQCAr0hNTdW2bdt0ySWXnHD/f/r2Gt8M4xwYfO7oGk4HgN8UFBTo5Zdf1rPPPnvC/TNnztTvfvc7HTt2LMCVhYYDBw6opqZGY8eOVXR0dLDL6XE+/fRT3XvvvXr66af13nvv6dixYxo0aJBGjRql2267TSNGjAh2iT3G8btUvPbaa7rgggs0b948rVu3Tjk5OTp69KgmTZqk4uJi/p0CQDfy8ssv68iRI/rpT396wv1HjhxRdXW10tPTA1xZaGGcA4PPHV1DCAAAAAAAgEVwTQAAAAAAACyCEAAAAAAAAIsgBAAAAAAAwCIIAQAAAAAAsAhCAAAAAAAALIIQAAAAAAAAiyAEAAAAAADAIv4/YMz3UtdIqGUAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 1200x400 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "messages.hist(column='length',by='label',bins=60,figsize=(12,4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "9d6143af",
   "metadata": {},
   "outputs": [],
   "source": [
    "mess='Sample Message! Notice: It has a punctuation.'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "fae3dcf7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import string"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "2e8fc7d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "nopunc = [c for c in mess if c not in string.punctuation]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "66b4abfc",
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.corpus import stopwords"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "0daaa771",
   "metadata": {},
   "outputs": [],
   "source": [
    "nopunc=''.join(nopunc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "2a967676",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Sample Message Notice It has a punctuation'"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nopunc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "b98042fb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Sample', 'Message', 'Notice', 'It', 'has', 'a', 'punctuation']"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nopunc.split()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "af1f749c",
   "metadata": {},
   "outputs": [],
   "source": [
    "clean_mess=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "ef05bfb8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Sample', 'Message', 'Notice', 'punctuation']"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clean_mess"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "65f87751",
   "metadata": {},
   "outputs": [],
   "source": [
    "def text_process(mess):\n",
    "    \"\"\"\n",
    "    Takes in a string of text, then performs the following:\n",
    "    1. Remove all punctuation\n",
    "    2. Remove all stopwords\n",
    "    3. Returns a list of the cleaned text\n",
    "    \"\"\"\n",
    "    # Check characters to see if they are in punctuation\n",
    "    nopunc = [char for char in mess if char not in string.punctuation]\n",
    "\n",
    "    # Join the characters again to form the string.\n",
    "    nopunc = ''.join(nopunc)\n",
    "    \n",
    "    # Now just remove any stopwords\n",
    "    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "f41f923c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    [Go, jurong, point, crazy, Available, bugis, n...\n",
       "1                       [Ok, lar, Joking, wif, u, oni]\n",
       "2    [Free, entry, 2, wkly, comp, win, FA, Cup, fin...\n",
       "3        [U, dun, say, early, hor, U, c, already, say]\n",
       "4    [Nah, dont, think, goes, usf, lives, around, t...\n",
       "Name: message, dtype: object"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages['message'].head(5).apply(text_process)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "ae03cd20",
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
       "      <th>label</th>\n",
       "      <th>message</th>\n",
       "      <th>length</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "      <td>111</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "      <td>29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "      <td>155</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "      <td>49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "      <td>61</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  label                                            message  length\n",
       "0   ham  Go until jurong point, crazy.. Available only ...     111\n",
       "1   ham                      Ok lar... Joking wif u oni...      29\n",
       "2  spam  Free entry in 2 a wkly comp to win FA Cup fina...     155\n",
       "3   ham  U dun say so early hor... U c already then say...      49\n",
       "4   ham  Nah I don't think he goes to usf, he lives aro...      61"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "d6aa02c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import CountVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "742aa1c2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "11425\n"
     ]
    }
   ],
   "source": [
    "# Might take awhile...\n",
    "bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])\n",
    "\n",
    "# Print total number of vocab words\n",
    "print(len(bow_transformer.vocabulary_))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "08484357",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "U dun say so early hor... U c already then say...\n"
     ]
    }
   ],
   "source": [
    "message4 = messages['message'][3]\n",
    "print(message4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "ef13a30e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  (0, 4068)\t2\n",
      "  (0, 4629)\t1\n",
      "  (0, 5261)\t1\n",
      "  (0, 6204)\t1\n",
      "  (0, 6222)\t1\n",
      "  (0, 7186)\t1\n",
      "  (0, 9554)\t2\n",
      "(1, 11425)\n"
     ]
    }
   ],
   "source": [
    "bow4 = bow_transformer.transform([message4])\n",
    "print(bow4)\n",
    "print(bow4.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "ce9e058f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "UIN\n",
      "schedule\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\DELL\\anaconda3\\lib\\site-packages\\sklearn\\utils\\deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.\n",
      "  warnings.warn(msg, category=FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "print(bow_transformer.get_feature_names()[4073])\n",
    "print(bow_transformer.get_feature_names()[9570])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "4f13bb5f",
   "metadata": {},
   "outputs": [],
   "source": [
    "messages_bow = bow_transformer.transform(messages['message'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "c604e1ed",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of Sparse Matrix:  (5572, 11425)\n",
      "Amount of Non-Zero occurences:  50548\n"
     ]
    }
   ],
   "source": [
    "print('Shape of Sparse Matrix: ', messages_bow.shape)\n",
    "print('Amount of Non-Zero occurences: ', messages_bow.nnz)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "50f34758",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sparsity: 0\n"
     ]
    }
   ],
   "source": [
    "sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))\n",
    "print('sparsity: {}'.format(round(sparsity)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "38952cc2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  (0, 9554)\t0.5385626262927564\n",
      "  (0, 7186)\t0.4389365653379857\n",
      "  (0, 6222)\t0.3187216892949149\n",
      "  (0, 6204)\t0.29953799723697416\n",
      "  (0, 5261)\t0.29729957405868723\n",
      "  (0, 4629)\t0.26619801906087187\n",
      "  (0, 4068)\t0.40832589933384067\n"
     ]
    }
   ],
   "source": [
    "from sklearn.feature_extraction.text import TfidfTransformer\n",
    "\n",
    "tfidf_transformer = TfidfTransformer().fit(messages_bow)\n",
    "tfidf4 = tfidf_transformer.transform(bow4)\n",
    "print(tfidf4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "3b52b15f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3.2800524267409408\n",
      "8.527076498901426\n"
     ]
    }
   ],
   "source": [
    "print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])\n",
    "print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "0a1b5062",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(5572, 11425)\n"
     ]
    }
   ],
   "source": [
    "messages_tfidf = tfidf_transformer.transform(messages_bow)\n",
    "print(messages_tfidf.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "ecfe6f8d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.naive_bayes import MultinomialNB\n",
    "spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "bb581f19",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "predicted: ham\n",
      "expected: ham\n"
     ]
    }
   ],
   "source": [
    "print('predicted:', spam_detect_model.predict(tfidf4)[0])\n",
    "print('expected:', messages.label[3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "7ea60534",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['ham' 'ham' 'spam' ... 'ham' 'ham' 'ham']\n"
     ]
    }
   ],
   "source": [
    "all_predictions = spam_detect_model.predict(messages_tfidf)\n",
    "print(all_predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "44cd4692",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         ham       0.98      1.00      0.99      4825\n",
      "        spam       1.00      0.85      0.92       747\n",
      "\n",
      "    accuracy                           0.98      5572\n",
      "   macro avg       0.99      0.92      0.95      5572\n",
      "weighted avg       0.98      0.98      0.98      5572\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print (classification_report(messages['label'], all_predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "1633b119",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "4457 1115 5572\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "msg_train, msg_test, label_train, label_test = \\\n",
    "train_test_split(messages['message'], messages['label'], test_size=0.2)\n",
    "\n",
    "print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "2dfcda1a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.pipeline import Pipeline\n",
    "\n",
    "pipeline = Pipeline([\n",
    "    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts\n",
    "    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores\n",
    "    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "f4d66501",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pipeline(steps=[('bow',\n",
       "                 CountVectorizer(analyzer=<function text_process at 0x000001533F011790>)),\n",
       "                ('tfidf', TfidfTransformer()),\n",
       "                ('classifier', MultinomialNB())])"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pipeline.fit(msg_train,label_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "79396b70",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions = pipeline.predict(msg_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "be3e24ec",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         ham       1.00      0.95      0.98      1005\n",
      "        spam       0.70      1.00      0.82       110\n",
      "\n",
      "    accuracy                           0.96      1115\n",
      "   macro avg       0.85      0.98      0.90      1115\n",
      "weighted avg       0.97      0.96      0.96      1115\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(predictions,label_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "103def8a",
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
