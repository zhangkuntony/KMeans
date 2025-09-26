import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def load_user_behavior_data():
    user_behavior = pd.read_csv('./data/UserBehavior-15k.csv')
    print(user_behavior.columns)
    user_behavior.columns = ['user', 'item', 'category', 'action', 'ts']
    print(user_behavior.columns)
    print(user_behavior.head())
    return user_behavior

def calc_user_action_count(user_behavior):
    user_action_count1 = pd.pivot_table(user_behavior[['user', 'action', 'item']],
                                        index=['user'], columns=['action'], aggfunc='count')
    print('user_action_count1')
    print(user_action_count1)

    user_action_count2 = user_behavior.groupby('user').agg({
        'item': ['count', 'nunique'],
        'action': ['nunique'],
        'ts': ['nunique', np.ptp]
    })

    print('user_action_count2')
    print(user_action_count2)

    user_action_count1 = user_action_count1.reset_index()
    user_action_count1.columns = [x[0]+x[1] for x in user_action_count1.columns]
    user_action_count1 = user_action_count1.fillna(0)

    print('user_action_count1.head')
    print(user_action_count1.head())

    print('user_action_count1.columns')
    print(user_action_count1.columns)

    user_action_count2 = user_action_count2.reset_index()
    user_action_count2.columns = [x[0]+x[1] for x in user_action_count2.columns]
    user_action_count2 = user_action_count2.fillna(0)

    print('user_action_count2.head')
    print(user_action_count2.head())

    print('user_action_count2.columns')
    print(user_action_count2.columns)

    user_action_count = pd.merge(user_action_count1, user_action_count2, on='user')
    for col in user_action_count.columns[1:]:
        user_action_count[col] /= user_action_count[col].max()

    print('user_action_count.head')
    print(user_action_count.head(10))
    return user_action_count

def user_behavior_kmeans():
    user_behavior_data = load_user_behavior_data()
    user_action_count_data = calc_user_action_count(user_behavior_data)

    kmeans = KMeans(
        init='random',
        n_clusters=3,
    )

    kmeans.fit(user_action_count_data.iloc[:, 1:])

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 500,
        "random_state": 42,
    }

    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(user_action_count_data.iloc[:, 1:])
        sse.append(kmeans.inertia_)

    plt.style.use('fivethirtyeight')
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()

if __name__ == '__main__':
    user_behavior_kmeans()