# KMeans Clustering Algorithm Implementation and Applications

This project demonstrates the implementation and applications of K-Means clustering algorithm in different scenarios, including basic clustering analysis, image segmentation, and user behavior analysis.

## Project Structure

```
KMeans/
├── src/                   # Source code directory
│   ├── KMeans.py          # Basic K-Means clustering implementation
│   ├── ImageKMeans.py     # Image segmentation application
│   └── UserBehaviorKMeans.py  # User behavior analysis application
├── data/                  # Data files
│   └── UserBehavior-15k.csv  # User behavior dataset
└── images/                # Image files
    └── dog-cat.jpeg       # Sample image for segmentation
```

## Functional Modules

### 1. Basic Clustering Analysis (src/KMeans.py)

Implements the basic functionality of K-Means clustering algorithm, including:

- Generating simulated clustering datasets
- Basic K-Means clustering implementation
- Using the Elbow Method to determine the optimal number of clusters
- Using Silhouette Coefficient to evaluate clustering quality

Main functions:
- `init_data()`: Generate simulated clustering data
- `kmeans_basic()`: Basic K-Means clustering implementation
- `kmeans_kwargs()`: Use the Elbow Method to determine the optimal K value
- `kmeans_silhouette_coefficient()`: Calculate Silhouette Coefficient to evaluate clustering quality

### 2. Image Segmentation (src/ImageKMeans.py)

Uses K-Means algorithm for image segmentation, clustering image pixels by color features to achieve image color quantization.

Main functions:
- `load_image()`: Load and display the original image
- `image_kmeans()`: Perform K-Means clustering segmentation on the image

### 3. User Behavior Analysis (src/UserBehaviorKMeans.py)

Analyzes user behavior data, using K-Means algorithm to segment users and discover different user behavior patterns.

Main functions:
- `load_user_behavior_data()`: Load user behavior data
- `calc_user_action_count()`: Calculate user behavior features
- `user_behavior_kmeans()`: Perform clustering analysis on user behavior

## Dataset

### UserBehavior-15k.csv

User behavior dataset containing the following fields:
- user: User ID
- item: Product ID
- category: Product category
- action: User action type
- ts: Timestamp

## Requirements

- Python 3.6+
- Dependencies:
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - opencv-python (cv2)

## Usage

### Basic Clustering Analysis

```bash
python src/KMeans.py
```

### Image Segmentation

```bash
python src/ImageKMeans.py
```

### User Behavior Analysis

```bash
python src/UserBehaviorKMeans.py
```

## Algorithm Description

### K-Means Clustering Algorithm

K-Means is a commonly used clustering algorithm with the following basic steps:

1. Randomly select K points as initial cluster centers
2. Assign each data point to the nearest cluster center
3. Recalculate the center point of each cluster
4. Repeat steps 2 and 3 until the cluster centers no longer change or the maximum number of iterations is reached

### Evaluation Metrics

- **Inertia (WCSS)**: The sum of squared distances of all data points to their respective cluster centers; lower values indicate better clustering
- **Silhouette Coefficient**: Measures the density and separation of clusters, with values ranging from [-1,1]; values closer to 1 indicate better clustering

## Application Scenarios

- **Data Analysis**: Discovering natural groupings and patterns in data
- **Image Processing**: Image segmentation, color quantization, feature extraction
- **User Profiling**: User segmentation, behavior analysis, personalized recommendations

## References

- [K-Means Clustering Algorithm Principles](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Silhouette Coefficient Evaluation Method](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [Elbow Method for Determining Optimal K Value](https://en.wikipedia.org/wiki/Elbow_method_(clustering))