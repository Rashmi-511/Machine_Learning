{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d5fa4e29",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "45f292d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_csv('PYTORCH_NOTEBOOKS/Data/NYCTaxiFares.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "94f4ebf3",
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
       "      <th>pickup_datetime</th>\n",
       "      <th>fare_amount</th>\n",
       "      <th>fare_class</th>\n",
       "      <th>pickup_longitude</th>\n",
       "      <th>pickup_latitude</th>\n",
       "      <th>dropoff_longitude</th>\n",
       "      <th>dropoff_latitude</th>\n",
       "      <th>passenger_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2010-04-19 08:17:56 UTC</td>\n",
       "      <td>6.5</td>\n",
       "      <td>0</td>\n",
       "      <td>-73.992365</td>\n",
       "      <td>40.730521</td>\n",
       "      <td>-73.975499</td>\n",
       "      <td>40.744746</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2010-04-17 15:43:53 UTC</td>\n",
       "      <td>6.9</td>\n",
       "      <td>0</td>\n",
       "      <td>-73.990078</td>\n",
       "      <td>40.740558</td>\n",
       "      <td>-73.974232</td>\n",
       "      <td>40.744114</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2010-04-17 11:23:26 UTC</td>\n",
       "      <td>10.1</td>\n",
       "      <td>1</td>\n",
       "      <td>-73.994149</td>\n",
       "      <td>40.751118</td>\n",
       "      <td>-73.960064</td>\n",
       "      <td>40.766235</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2010-04-11 21:25:03 UTC</td>\n",
       "      <td>8.9</td>\n",
       "      <td>0</td>\n",
       "      <td>-73.990485</td>\n",
       "      <td>40.756422</td>\n",
       "      <td>-73.971205</td>\n",
       "      <td>40.748192</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2010-04-17 02:19:01 UTC</td>\n",
       "      <td>19.7</td>\n",
       "      <td>1</td>\n",
       "      <td>-73.990976</td>\n",
       "      <td>40.734202</td>\n",
       "      <td>-73.905956</td>\n",
       "      <td>40.743115</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           pickup_datetime  fare_amount  fare_class  pickup_longitude  \\\n",
       "0  2010-04-19 08:17:56 UTC          6.5           0        -73.992365   \n",
       "1  2010-04-17 15:43:53 UTC          6.9           0        -73.990078   \n",
       "2  2010-04-17 11:23:26 UTC         10.1           1        -73.994149   \n",
       "3  2010-04-11 21:25:03 UTC          8.9           0        -73.990485   \n",
       "4  2010-04-17 02:19:01 UTC         19.7           1        -73.990976   \n",
       "\n",
       "   pickup_latitude  dropoff_longitude  dropoff_latitude  passenger_count  \n",
       "0        40.730521         -73.975499         40.744746                1  \n",
       "1        40.740558         -73.974232         40.744114                1  \n",
       "2        40.751118         -73.960064         40.766235                2  \n",
       "3        40.756422         -73.971205         40.748192                1  \n",
       "4        40.734202         -73.905956         40.743115                1  "
      ]
     },
     "execution_count": 4,
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
   "execution_count": 5,
   "id": "65be4b6a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    120000.000000\n",
       "mean         10.040326\n",
       "std           7.500134\n",
       "min           2.500000\n",
       "25%           5.700000\n",
       "50%           7.700000\n",
       "75%          11.300000\n",
       "max          49.900000\n",
       "Name: fare_amount, dtype: float64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['fare_amount'].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "ea87aff6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def haversine_distance(df, lat1, long1, lat2, long2):\n",
    "    \"\"\"\n",
    "    Calculates the haversine distance between 2 sets of GPS coordinates in df\n",
    "    \"\"\"\n",
    "    r = 6371  # average radius of Earth in kilometers\n",
    "       \n",
    "    phi1 = np.radians(df[lat1])\n",
    "    phi2 = np.radians(df[lat2])\n",
    "    \n",
    "    delta_phi = np.radians(df[lat2]-df[lat1])\n",
    "    delta_lambda = np.radians(df[long2]-df[long1])\n",
    "     \n",
    "    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2\n",
    "    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))\n",
    "    d = (r * c) # in kilometers\n",
    "\n",
    "    return d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "57fbd884",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['dist_km']=haversine_distance(df,'pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "0dbb36f1",
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
       "      <th>pickup_datetime</th>\n",
       "      <th>fare_amount</th>\n",
       "      <th>fare_class</th>\n",
       "      <th>pickup_longitude</th>\n",
       "      <th>pickup_latitude</th>\n",
       "      <th>dropoff_longitude</th>\n",
       "      <th>dropoff_latitude</th>\n",
       "      <th>passenger_count</th>\n",
       "      <th>dist_km</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2010-04-19 08:17:56 UTC</td>\n",
       "      <td>6.5</td>\n",
       "      <td>0</td>\n",
       "      <td>-73.992365</td>\n",
       "      <td>40.730521</td>\n",
       "      <td>-73.975499</td>\n",
       "      <td>40.744746</td>\n",
       "      <td>1</td>\n",
       "      <td>2.126312</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2010-04-17 15:43:53 UTC</td>\n",
       "      <td>6.9</td>\n",
       "      <td>0</td>\n",
       "      <td>-73.990078</td>\n",
       "      <td>40.740558</td>\n",
       "      <td>-73.974232</td>\n",
       "      <td>40.744114</td>\n",
       "      <td>1</td>\n",
       "      <td>1.392307</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2010-04-17 11:23:26 UTC</td>\n",
       "      <td>10.1</td>\n",
       "      <td>1</td>\n",
       "      <td>-73.994149</td>\n",
       "      <td>40.751118</td>\n",
       "      <td>-73.960064</td>\n",
       "      <td>40.766235</td>\n",
       "      <td>2</td>\n",
       "      <td>3.326763</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2010-04-11 21:25:03 UTC</td>\n",
       "      <td>8.9</td>\n",
       "      <td>0</td>\n",
       "      <td>-73.990485</td>\n",
       "      <td>40.756422</td>\n",
       "      <td>-73.971205</td>\n",
       "      <td>40.748192</td>\n",
       "      <td>1</td>\n",
       "      <td>1.864129</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2010-04-17 02:19:01 UTC</td>\n",
       "      <td>19.7</td>\n",
       "      <td>1</td>\n",
       "      <td>-73.990976</td>\n",
       "      <td>40.734202</td>\n",
       "      <td>-73.905956</td>\n",
       "      <td>40.743115</td>\n",
       "      <td>1</td>\n",
       "      <td>7.231321</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           pickup_datetime  fare_amount  fare_class  pickup_longitude  \\\n",
       "0  2010-04-19 08:17:56 UTC          6.5           0        -73.992365   \n",
       "1  2010-04-17 15:43:53 UTC          6.9           0        -73.990078   \n",
       "2  2010-04-17 11:23:26 UTC         10.1           1        -73.994149   \n",
       "3  2010-04-11 21:25:03 UTC          8.9           0        -73.990485   \n",
       "4  2010-04-17 02:19:01 UTC         19.7           1        -73.990976   \n",
       "\n",
       "   pickup_latitude  dropoff_longitude  dropoff_latitude  passenger_count  \\\n",
       "0        40.730521         -73.975499         40.744746                1   \n",
       "1        40.740558         -73.974232         40.744114                1   \n",
       "2        40.751118         -73.960064         40.766235                2   \n",
       "3        40.756422         -73.971205         40.748192                1   \n",
       "4        40.734202         -73.905956         40.743115                1   \n",
       "\n",
       "    dist_km  \n",
       "0  2.126312  \n",
       "1  1.392307  \n",
       "2  3.326763  \n",
       "3  1.864129  \n",
       "4  7.231321  "
      ]
     },
     "execution_count": 9,
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
   "id": "05f6e4ca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 120000 entries, 0 to 119999\n",
      "Data columns (total 9 columns):\n",
      " #   Column             Non-Null Count   Dtype  \n",
      "---  ------             --------------   -----  \n",
      " 0   pickup_datetime    120000 non-null  object \n",
      " 1   fare_amount        120000 non-null  float64\n",
      " 2   fare_class         120000 non-null  int64  \n",
      " 3   pickup_longitude   120000 non-null  float64\n",
      " 4   pickup_latitude    120000 non-null  float64\n",
      " 5   dropoff_longitude  120000 non-null  float64\n",
      " 6   dropoff_latitude   120000 non-null  float64\n",
      " 7   passenger_count    120000 non-null  int64  \n",
      " 8   dist_km            120000 non-null  float64\n",
      "dtypes: float64(6), int64(2), object(1)\n",
      "memory usage: 8.2+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e95c29c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "b9c5a0d4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 120000 entries, 0 to 119999\n",
      "Data columns (total 9 columns):\n",
      " #   Column             Non-Null Count   Dtype              \n",
      "---  ------             --------------   -----              \n",
      " 0   pickup_datetime    120000 non-null  datetime64[ns, UTC]\n",
      " 1   fare_amount        120000 non-null  float64            \n",
      " 2   fare_class         120000 non-null  int64              \n",
      " 3   pickup_longitude   120000 non-null  float64            \n",
      " 4   pickup_latitude    120000 non-null  float64            \n",
      " 5   dropoff_longitude  120000 non-null  float64            \n",
      " 6   dropoff_latitude   120000 non-null  float64            \n",
      " 7   passenger_count    120000 non-null  int64              \n",
      " 8   dist_km            120000 non-null  float64            \n",
      "dtypes: datetime64[ns, UTC](1), float64(6), int64(2)\n",
      "memory usage: 8.2 MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "3448da3c",
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
       "      <th>pickup_datetime</th>\n",
       "      <th>fare_amount</th>\n",
       "      <th>fare_class</th>\n",
       "      <th>pickup_longitude</th>\n",
       "      <th>pickup_latitude</th>\n",
       "      <th>dropoff_longitude</th>\n",
       "      <th>dropoff_latitude</th>\n",
       "      <th>passenger_count</th>\n",
       "      <th>dist_km</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2010-04-19 08:17:56+00:00</td>\n",
       "      <td>6.5</td>\n",
       "      <td>0</td>\n",
       "      <td>-73.992365</td>\n",
       "      <td>40.730521</td>\n",
       "      <td>-73.975499</td>\n",
       "      <td>40.744746</td>\n",
       "      <td>1</td>\n",
       "      <td>2.126312</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2010-04-17 15:43:53+00:00</td>\n",
       "      <td>6.9</td>\n",
       "      <td>0</td>\n",
       "      <td>-73.990078</td>\n",
       "      <td>40.740558</td>\n",
       "      <td>-73.974232</td>\n",
       "      <td>40.744114</td>\n",
       "      <td>1</td>\n",
       "      <td>1.392307</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2010-04-17 11:23:26+00:00</td>\n",
       "      <td>10.1</td>\n",
       "      <td>1</td>\n",
       "      <td>-73.994149</td>\n",
       "      <td>40.751118</td>\n",
       "      <td>-73.960064</td>\n",
       "      <td>40.766235</td>\n",
       "      <td>2</td>\n",
       "      <td>3.326763</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2010-04-11 21:25:03+00:00</td>\n",
       "      <td>8.9</td>\n",
       "      <td>0</td>\n",
       "      <td>-73.990485</td>\n",
       "      <td>40.756422</td>\n",
       "      <td>-73.971205</td>\n",
       "      <td>40.748192</td>\n",
       "      <td>1</td>\n",
       "      <td>1.864129</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2010-04-17 02:19:01+00:00</td>\n",
       "      <td>19.7</td>\n",
       "      <td>1</td>\n",
       "      <td>-73.990976</td>\n",
       "      <td>40.734202</td>\n",
       "      <td>-73.905956</td>\n",
       "      <td>40.743115</td>\n",
       "      <td>1</td>\n",
       "      <td>7.231321</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            pickup_datetime  fare_amount  fare_class  pickup_longitude  \\\n",
       "0 2010-04-19 08:17:56+00:00          6.5           0        -73.992365   \n",
       "1 2010-04-17 15:43:53+00:00          6.9           0        -73.990078   \n",
       "2 2010-04-17 11:23:26+00:00         10.1           1        -73.994149   \n",
       "3 2010-04-11 21:25:03+00:00          8.9           0        -73.990485   \n",
       "4 2010-04-17 02:19:01+00:00         19.7           1        -73.990976   \n",
       "\n",
       "   pickup_latitude  dropoff_longitude  dropoff_latitude  passenger_count  \\\n",
       "0        40.730521         -73.975499         40.744746                1   \n",
       "1        40.740558         -73.974232         40.744114                1   \n",
       "2        40.751118         -73.960064         40.766235                2   \n",
       "3        40.756422         -73.971205         40.748192                1   \n",
       "4        40.734202         -73.905956         40.743115                1   \n",
       "\n",
       "    dist_km  \n",
       "0  2.126312  \n",
       "1  1.392307  \n",
       "2  3.326763  \n",
       "3  1.864129  \n",
       "4  7.231321  "
      ]
     },
     "execution_count": 13,
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
   "execution_count": 14,
   "id": "be9eea60",
   "metadata": {},
   "outputs": [],
   "source": [
    "my_time=df['pickup_datetime'][0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "b7caebec",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Timestamp('2010-04-19 08:17:56+0000', tz='UTC')"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "my_time.hour"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "6870c2bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['ETDdate']=df['pickup_datetime']-pd.Timedelta(hours=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "23e78745",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Hour']=df['ETDdate'].dt.hour"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "ea142c7f",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['AMorPM']=np.where(df['Hour']<12,'am','pm')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "df5cc23d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['pickup_datetime', 'fare_amount', 'fare_class', 'pickup_longitude',\n",
       "       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',\n",
       "       'passenger_count', 'dist_km', 'ETDdate', 'Hour', 'AMorPM', 'Weekday'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "ee59a171",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Weekday']=df['ETDdate'].dt.strftime(\"%a\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "bb0800eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "cat_cols=['Hour','AMorPM','Weekday']\n",
    "cont_cols=['pickup_longitude',\n",
    "       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',\n",
    "       'passenger_count', 'dist_km']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "7a8673cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_col=['fare_amount']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "e3c898ee",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "pickup_datetime      datetime64[ns, UTC]\n",
       "fare_amount                      float64\n",
       "fare_class                         int64\n",
       "pickup_longitude                 float64\n",
       "pickup_latitude                  float64\n",
       "dropoff_longitude                float64\n",
       "dropoff_latitude                 float64\n",
       "passenger_count                    int64\n",
       "dist_km                          float64\n",
       "ETDdate              datetime64[ns, UTC]\n",
       "Hour                            category\n",
       "AMorPM                          category\n",
       "Weekday                         category\n",
       "dtype: object"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "0cf9147c",
   "metadata": {},
   "outputs": [],
   "source": [
    "for cat in cat_cols:\n",
    "    df[cat]=df[cat].astype('category')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "9332f50b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     4\n",
       "1    11\n",
       "2     7\n",
       "3    17\n",
       "4    22\n",
       "Name: Hour, dtype: category\n",
       "Categories (24, int64): [0, 1, 2, 3, ..., 20, 21, 22, 23]"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Hour'].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "161cd970",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    Mon\n",
       "1    Sat\n",
       "2    Sat\n",
       "3    Sun\n",
       "4    Fri\n",
       "Name: Weekday, dtype: category\n",
       "Categories (7, object): ['Fri', 'Mon', 'Sat', 'Sun', 'Thu', 'Tue', 'Wed']"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Weekday'].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "997d1c07",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    am\n",
       "1    am\n",
       "2    am\n",
       "3    pm\n",
       "4    pm\n",
       "Name: AMorPM, dtype: category\n",
       "Categories (2, object): ['am', 'pm']"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['AMorPM'].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "4c0fb89a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 2, ..., 3, 5, 2], dtype=int8)"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Weekday'].cat.codes.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "ec84de89",
   "metadata": {},
   "outputs": [],
   "source": [
    "hr=df['Hour'].cat.codes.values\n",
    "ampm=df['AMorPM'].cat.codes.values\n",
    "wkdy=df['Weekday'].cat.codes.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "6ecb4655",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 4, 11,  7, ..., 14,  4, 12], dtype=int8)"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "hr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "fbeeef18",
   "metadata": {},
   "outputs": [],
   "source": [
    "cats=np.stack([df[col].cat.codes.values for col in cat_cols],1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "9f993a63",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 4,  0,  1],\n",
       "       [11,  0,  2],\n",
       "       [ 7,  0,  2],\n",
       "       ...,\n",
       "       [14,  1,  3],\n",
       "       [ 4,  0,  5],\n",
       "       [12,  1,  2]], dtype=int8)"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cats"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "7f9a31f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "cats=torch.tensor(cats,dtype=torch.int64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "dcd41a7e",
   "metadata": {},
   "outputs": [],
   "source": [
    "conts=np.stack([df[col] for col in cont_cols],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "7fe488cf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-73.992365  ,  40.730521  , -73.975499  ,  40.744746  ,\n",
       "          1.        ,   2.12631159],\n",
       "       [-73.990078  ,  40.740558  , -73.974232  ,  40.744114  ,\n",
       "          1.        ,   1.39230687],\n",
       "       [-73.994149  ,  40.751118  , -73.960064  ,  40.766235  ,\n",
       "          2.        ,   3.32676344],\n",
       "       ...,\n",
       "       [-73.988574  ,  40.749772  , -74.011541  ,  40.707799  ,\n",
       "          3.        ,   5.05252282],\n",
       "       [-74.004449  ,  40.724529  , -73.992697  ,  40.730765  ,\n",
       "          1.        ,   1.20892296],\n",
       "       [-73.955415  ,  40.77192   , -73.967623  ,  40.763015  ,\n",
       "          3.        ,   1.42739869]])"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "conts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "87758f6e",
   "metadata": {},
   "outputs": [],
   "source": [
    "conts=torch.tensor(conts,dtype=torch.float64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "bbc49cb1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[-73.9924,  40.7305, -73.9755,  40.7447,   1.0000,   2.1263],\n",
       "        [-73.9901,  40.7406, -73.9742,  40.7441,   1.0000,   1.3923],\n",
       "        [-73.9941,  40.7511, -73.9601,  40.7662,   2.0000,   3.3268],\n",
       "        ...,\n",
       "        [-73.9886,  40.7498, -74.0115,  40.7078,   3.0000,   5.0525],\n",
       "        [-74.0044,  40.7245, -73.9927,  40.7308,   1.0000,   1.2089],\n",
       "        [-73.9554,  40.7719, -73.9676,  40.7630,   3.0000,   1.4274]],\n",
       "       dtype=torch.float64)"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "conts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "a6a46b53",
   "metadata": {},
   "outputs": [],
   "source": [
    "y=torch.tensor(df[y_col].values,dtype=torch.float64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "316acde1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([120000, 3])"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cats.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "029d2b98",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([120000, 6])"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "conts.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "c338c848",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([120000, 1])"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "59e91edd",
   "metadata": {},
   "outputs": [],
   "source": [
    "cats_szs=[len(df[col].cat.categories) for col in cat_cols]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "f67bd3a9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[24, 2, 7]"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cats_szs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "ded06ece",
   "metadata": {},
   "outputs": [],
   "source": [
    "emb_szs=[(size,min(50,(size+1)//2)) for size in cats_szs]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "678ac33d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[(24, 12), (2, 1), (7, 4)]"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "emb_szs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "d89134eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "catz=cats[:4]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "04ac9ece",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[ 4,  0,  1],\n",
       "        [11,  0,  2],\n",
       "        [ 7,  0,  2],\n",
       "        [17,  1,  3]])"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "catz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "4efda220",
   "metadata": {},
   "outputs": [],
   "source": [
    "selfembeds=nn.ModuleList([nn.Embedding(ni,nf) for ni,nf in emb_szs])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "379c0b7b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ModuleList(\n",
       "  (0): Embedding(24, 12)\n",
       "  (1): Embedding(2, 1)\n",
       "  (2): Embedding(7, 4)\n",
       ")"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "selfEmbeds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "ad199c56",
   "metadata": {},
   "outputs": [],
   "source": [
    "#FORWARD METHOD(cats)\n",
    "embeddings = []\n",
    "\n",
    "for i,e in enumerate(selfembeds):\n",
    "    embeddings.append(e(catz[:,i]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "c46d978f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[tensor([[ 1.5307,  0.9800, -1.4486,  2.0411,  0.4846,  1.3880, -1.1209,  0.0219,\n",
       "          -1.2238, -0.6941,  0.1130, -1.9026],\n",
       "         [ 1.1782, -0.3068,  0.1364, -0.3396,  0.3412, -0.3752, -1.4501, -0.6074,\n",
       "           1.2685, -0.4764,  2.4483, -0.1581],\n",
       "         [-0.1127, -0.1457, -0.3765,  0.1366, -0.9406,  0.5790,  1.1657,  1.2005,\n",
       "          -1.3632,  0.6568, -0.6431,  0.5171],\n",
       "         [ 0.5797,  0.6890,  0.8127, -0.3548, -0.5068, -1.1459, -0.7740,  0.7747,\n",
       "          -1.6812,  0.7435,  1.2391, -0.1114]], grad_fn=<EmbeddingBackward0>),\n",
       " tensor([[ 0.3629],\n",
       "         [ 0.3629],\n",
       "         [ 0.3629],\n",
       "         [-1.1666]], grad_fn=<EmbeddingBackward0>),\n",
       " tensor([[ 0.0566,  0.3526,  0.0198,  0.9061],\n",
       "         [-0.1203,  1.4714, -0.5815, -1.3003],\n",
       "         [-0.1203,  1.4714, -0.5815, -1.3003],\n",
       "         [ 1.5102, -1.0806,  0.6940,  0.9124]], grad_fn=<EmbeddingBackward0>)]"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "embeddings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "d887f17c",
   "metadata": {},
   "outputs": [],
   "source": [
    "z=torch.cat(embeddings,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "b269ee3a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[ 1.5307,  0.9800, -1.4486,  2.0411,  0.4846,  1.3880, -1.1209,  0.0219,\n",
       "         -1.2238, -0.6941,  0.1130, -1.9026,  0.3629,  0.0566,  0.3526,  0.0198,\n",
       "          0.9061],\n",
       "        [ 1.1782, -0.3068,  0.1364, -0.3396,  0.3412, -0.3752, -1.4501, -0.6074,\n",
       "          1.2685, -0.4764,  2.4483, -0.1581,  0.3629, -0.1203,  1.4714, -0.5815,\n",
       "         -1.3003],\n",
       "        [-0.1127, -0.1457, -0.3765,  0.1366, -0.9406,  0.5790,  1.1657,  1.2005,\n",
       "         -1.3632,  0.6568, -0.6431,  0.5171,  0.3629, -0.1203,  1.4714, -0.5815,\n",
       "         -1.3003],\n",
       "        [ 0.5797,  0.6890,  0.8127, -0.3548, -0.5068, -1.1459, -0.7740,  0.7747,\n",
       "         -1.6812,  0.7435,  1.2391, -0.1114, -1.1666,  1.5102, -1.0806,  0.6940,\n",
       "          0.9124]], grad_fn=<CatBackward0>)"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "z"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "8594c3cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "selfembdrop=nn.Dropout(0.4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "a8fa9644",
   "metadata": {},
   "outputs": [],
   "source": [
    "z=selfembdrop(z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "f9dd1ef1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[ 0.0000,  1.6333, -2.4143,  3.4018,  0.8077,  0.0000, -1.8681,  0.0000,\n",
       "         -2.0397, -1.1568,  0.1884, -0.0000,  0.6048,  0.0943,  0.5877,  0.0331,\n",
       "          1.5101],\n",
       "        [ 1.9636, -0.5113,  0.0000, -0.5661,  0.5686, -0.6253, -2.4168, -0.0000,\n",
       "          0.0000, -0.0000,  4.0805, -0.2635,  0.0000, -0.2005,  0.0000, -0.9692,\n",
       "         -0.0000],\n",
       "        [-0.0000, -0.0000, -0.0000,  0.2276, -0.0000,  0.0000,  0.0000,  2.0008,\n",
       "         -0.0000,  0.0000, -1.0718,  0.0000,  0.0000, -0.0000,  0.0000, -0.9692,\n",
       "         -2.1672],\n",
       "        [ 0.0000,  0.0000,  1.3545, -0.5913, -0.8447, -1.9099, -1.2901,  1.2911,\n",
       "         -2.8019,  1.2392,  2.0651, -0.1857, -1.9443,  2.5170, -1.8011,  1.1566,\n",
       "          1.5206]], grad_fn=<MulBackward0>)"
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "z"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "51ebe0aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "class TabularModule(nn.Module):\n",
    "    def __init__(self,emb_szs,n_cont,out_sz,layers,p=0.5):\n",
    "        \n",
    "        super().__init__()\n",
    "        self.embeds=nn.ModuleList([nn.Embedding(ni,nf) for ni,nf in emb_szs])\n",
    "        self.emb_drop=nn.Dropout(p)\n",
    "        self.bn_cont=nn.BatchNorm1d(n_cont)\n",
    "        layerlist=[]\n",
    "        n_emb=sum([nf for ni,nf in emb_szs])\n",
    "        n_in=n_emb+n_cont\n",
    "        \n",
    "        for i in layers:\n",
    "            layerlist.append(nn.Linear(n_in,i))\n",
    "            layerlist.append(nn.ReLU(inplace=True))\n",
    "            layerlist.append(nn.BatchNorm1d(i))\n",
    "            layerlist.append(nn.Dropout(p))\n",
    "            n_in=i\n",
    "        \n",
    "        layerlist.append(nn.Linear(layers[-1],out_sz))\n",
    "        self.layers=nn.Sequential(*layerlist)\n",
    "        \n",
    "    def forward(self,x_cat,x_cont):\n",
    "        embeddings=[]\n",
    "        \n",
    "        for i,e in enumerate(self.embeds):\n",
    "            embeddings.append(e(x_cat[:,i]))\n",
    "            \n",
    "        x=torch.cat(embeddings,1)\n",
    "        x=self.emb_drop(x)\n",
    "        \n",
    "        x_cont=self.bn_cont(x_cont)\n",
    "        x=torch.cat([x,x_cont],1)\n",
    "        x=self.layers(x)\n",
    "        return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "a1a2fe03",
   "metadata": {},
   "outputs": [],
   "source": [
    "torch.manual_seed(33)\n",
    "model=TabularModule(emb_szs,conts.shape[1],1,[200,100],p=0.4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "id": "03a43594",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "TabularModule(\n",
       "  (embeds): ModuleList(\n",
       "    (0): Embedding(24, 12)\n",
       "    (1): Embedding(2, 1)\n",
       "    (2): Embedding(7, 4)\n",
       "  )\n",
       "  (emb_drop): Dropout(p=0.4, inplace=False)\n",
       "  (bn_cont): BatchNorm1d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "  (layers): Sequential(\n",
       "    (0): Linear(in_features=23, out_features=200, bias=True)\n",
       "    (1): ReLU(inplace=True)\n",
       "    (2): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (3): Dropout(p=0.4, inplace=False)\n",
       "    (4): Linear(in_features=200, out_features=100, bias=True)\n",
       "    (5): ReLU(inplace=True)\n",
       "    (6): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (7): Dropout(p=0.4, inplace=False)\n",
       "    (8): Linear(in_features=100, out_features=1, bias=True)\n",
       "  )\n",
       ")"
      ]
     },
     "execution_count": 108,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "id": "ac7143a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "criterion=nn.MSELoss()\n",
    "optimizer=torch.optim.Adam(model.parameters(),lr=0.001)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "6b9f6879",
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size=60000\n",
    "test_size=int(batch_size*.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "id": "6b093d04",
   "metadata": {},
   "outputs": [],
   "source": [
    "cat_train=cats[:batch_size-test_size]\n",
    "cat_test=cats[batch_size-test_size:batch_size]\n",
    "con_train=conts[:batch_size-test_size]\n",
    "con_test=conts[batch_size-test_size:batch_size]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "id": "0e8aaf74",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train=y[:batch_size-test_size]\n",
    "y_test=y[batch_size-test_size:batch_size]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "id": "067b4ee2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "48000"
      ]
     },
     "execution_count": 120,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(cat_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "id": "3b562933",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[ 4,  0,  1],\n",
       "        [11,  0,  2],\n",
       "        [ 7,  0,  2],\n",
       "        ...,\n",
       "        [ 0,  0,  0],\n",
       "        [ 8,  0,  4],\n",
       "        [23,  1,  2]])"
      ]
     },
     "execution_count": 124,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cat_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "id": "612a9123",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[ 6],\n",
       "        [ 6],\n",
       "        [10],\n",
       "        ...,\n",
       "        [ 3],\n",
       "        [ 6],\n",
       "        [18]], dtype=torch.int32)"
      ]
     },
     "execution_count": 136,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train.int()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "id": "8a156657",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "48000"
      ]
     },
     "execution_count": 121,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(con_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "id": "bf4da324",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "12000"
      ]
     },
     "execution_count": 122,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(cat_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "id": "c4568c03",
   "metadata": {},
   "outputs": [
    {
     "ename": "RuntimeError",
     "evalue": "mixed dtype (CPU): expect input to have scalar type of BFloat16",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp\\ipykernel_13628\\1377214674.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      7\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mepochs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      8\u001b[0m     \u001b[0mi\u001b[0m\u001b[1;33m+=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 9\u001b[1;33m     \u001b[0my_pred\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcat_train\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcon_train\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     10\u001b[0m     \u001b[0mloss\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msqrt\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcriterion\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my_pred\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;31m# RMSE\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     11\u001b[0m     \u001b[0mlosses\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mloss\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\torch\\nn\\modules\\module.py\u001b[0m in \u001b[0;36m_call_impl\u001b[1;34m(self, *input, **kwargs)\u001b[0m\n\u001b[0;32m   1128\u001b[0m         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks\n\u001b[0;32m   1129\u001b[0m                 or _global_forward_hooks or _global_forward_pre_hooks):\n\u001b[1;32m-> 1130\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mforward_call\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0minput\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1131\u001b[0m         \u001b[1;31m# Do not call functions when jit is used\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1132\u001b[0m         \u001b[0mfull_backward_hooks\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnon_full_backward_hooks\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m[\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Local\\Temp\\ipykernel_13628\\874839158.py\u001b[0m in \u001b[0;36mforward\u001b[1;34m(self, x_cat, x_cont)\u001b[0m\n\u001b[0;32m     29\u001b[0m         \u001b[0mx\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0memb_drop\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     30\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 31\u001b[1;33m         \u001b[0mx_cont\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mbn_cont\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx_cont\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     32\u001b[0m         \u001b[0mx\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mtorch\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mx_cont\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     33\u001b[0m         \u001b[0mx\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mlayers\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\torch\\nn\\modules\\module.py\u001b[0m in \u001b[0;36m_call_impl\u001b[1;34m(self, *input, **kwargs)\u001b[0m\n\u001b[0;32m   1128\u001b[0m         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks\n\u001b[0;32m   1129\u001b[0m                 or _global_forward_hooks or _global_forward_pre_hooks):\n\u001b[1;32m-> 1130\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mforward_call\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0minput\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1131\u001b[0m         \u001b[1;31m# Do not call functions when jit is used\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1132\u001b[0m         \u001b[0mfull_backward_hooks\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnon_full_backward_hooks\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m[\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\torch\\nn\\modules\\batchnorm.py\u001b[0m in \u001b[0;36mforward\u001b[1;34m(self, input)\u001b[0m\n\u001b[0;32m    166\u001b[0m         \u001b[0mused\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mnormalization\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mi\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0me\u001b[0m\u001b[1;33m.\u001b[0m \u001b[1;32min\u001b[0m \u001b[0meval\u001b[0m \u001b[0mmode\u001b[0m \u001b[0mwhen\u001b[0m \u001b[0mbuffers\u001b[0m \u001b[0mare\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    167\u001b[0m         \"\"\"\n\u001b[1;32m--> 168\u001b[1;33m         return F.batch_norm(\n\u001b[0m\u001b[0;32m    169\u001b[0m             \u001b[0minput\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    170\u001b[0m             \u001b[1;31m# If buffers are not to be tracked, ensure that they won't be updated\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\torch\\nn\\functional.py\u001b[0m in \u001b[0;36mbatch_norm\u001b[1;34m(input, running_mean, running_var, weight, bias, training, momentum, eps)\u001b[0m\n\u001b[0;32m   2436\u001b[0m         \u001b[0m_verify_batch_size\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0minput\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msize\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2437\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2438\u001b[1;33m     return torch.batch_norm(\n\u001b[0m\u001b[0;32m   2439\u001b[0m         \u001b[0minput\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mweight\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mbias\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrunning_mean\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrunning_var\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtraining\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmomentum\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0meps\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtorch\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mbackends\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcudnn\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0menabled\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2440\u001b[0m     )\n",
      "\u001b[1;31mRuntimeError\u001b[0m: mixed dtype (CPU): expect input to have scalar type of BFloat16"
     ]
    }
   ],
   "source": [
    "import time\n",
    "start_time = time.time()\n",
    "\n",
    "epochs = 300\n",
    "losses = []\n",
    "\n",
    "for i in range(epochs):\n",
    "    i+=1\n",
    "    y_pred = model(cat_train, con_train)\n",
    "    loss = torch.sqrt(criterion(y_pred, y_train.int())) # RMSE\n",
    "    losses.append(loss)\n",
    "    \n",
    "    # a neat trick to save screen space:\n",
    "    if i%25 == 1:\n",
    "        print(f'epoch: {i}  loss: {loss}')\n",
    "\n",
    "    optimizer.zero_grad()\n",
    "    loss.backward()\n",
    "    optimizer.step()\n",
    "\n",
    "duration=time.time()-start_time\n",
    "print(f'Training took {duration/60} seconds')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "22ee728f",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.plot(range(epochs), losses)\n",
    "plt.ylabel('RMSE Loss')\n",
    "plt.xlabel('epoch');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7ebe12f5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# TO EVALUATE THE ENTIRE TEST SET\n",
    "with torch.no_grad():\n",
    "    y_val = model(cat_test, con_test)\n",
    "    loss = torch.sqrt(criterion(y_val, y_test))\n",
    "print(f'RMSE: {loss:.8f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "444dffb9",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f'{\"PREDICTED\":>12} {\"ACTUAL\":>8} {\"DIFF\":>8}')\n",
    "for i in range(50):\n",
    "    diff = np.abs(y_val[i].item()-y_test[i].item())\n",
    "    print(f'{i+1:2}. {y_val[i].item():8.4f} {y_test[i].item():8.4f} {diff:8.4f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fdc843b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Make sure to save the model only after the training has happened!\n",
    "if len(losses) == epochs:\n",
    "    torch.save(model.state_dict(), 'TaxiFareRegrModel.pt')\n",
    "else:\n",
    "    print('Model has not been trained. Consider loading a trained model instead.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3dd385d7",
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
