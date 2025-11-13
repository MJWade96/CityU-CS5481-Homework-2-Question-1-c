import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# ---------- Load Data ----------
# 使用正确的文件路径
data_dir = "ml-1m"
ratings_path = os.path.join(data_dir, "ratings.dat")
movies_path = os.path.join(data_dir, "movies.dat")
users_path = os.path.join(data_dir, "users.dat")

# 添加异常处理，检查文件是否存在
try:
    for file_path in [ratings_path, movies_path, users_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 读取文件时指定编码格式为latin-1
    ratings = pd.read_csv(ratings_path, sep="::", engine="python", 
                          names=["UserID", "MovieID", "Rating", "Timestamp"], encoding='latin-1')
    movies = pd.read_csv(movies_path, sep="::", engine="python", 
                         names=["MovieID", "Title", "Genres"], encoding='latin-1')
    users = pd.read_csv(users_path, sep="::", engine="python", 
                        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"], encoding='latin-1')
    
    # ---------- Handle Missing Values ----------
    # Check for missing
    print("Missing in ratings:\n", ratings.isnull().sum())
    print("Missing in movies:\n", movies.isnull().sum())
    print("Missing in users:\n", users.isnull().sum())
    
    # Typical handling strategy:
    # - ratings: drop missing (since ratings are critical)
    # - movies: fill missing Genres with 'Unknown'
    # - users: fill missing Zip-code with mode, others drop or fill
    ratings.dropna(inplace=True)
    movies['Genres'].fillna('Unknown', inplace=True)
    users['Zip-code'].fillna(users['Zip-code'].mode()[0], inplace=True)
    
    # ---------- Encode Categorical Variables ----------
    ## (a) Movies - Genres
    # Split multi-genre field into multiple binary columns
    genre_dummies = movies['Genres'].str.get_dummies(sep='|')
    movies = pd.concat([movies.drop('Genres', axis=1), genre_dummies], axis=1)
    
    ## (b) Users - Gender
    gender_le = LabelEncoder()
    users['Gender'] = gender_le.fit_transform(users['Gender'])
    # (M=1, F=0)
    
    ## (c) Users - Occupation
    # Use OneHotEncoding because occupation is nominal, not ordinal
    occupation_ohe = pd.get_dummies(users['Occupation'], prefix='Occ')
    users = pd.concat([users.drop('Occupation', axis=1), occupation_ohe], axis=1)
    
    ## (d) Users - Zip-code
    # Option 1: drop Zip-code (too granular)
    # Option 2: encode first 3 digits as region
    users['ZipPrefix'] = users['Zip-code'].astype(str).str[:3]
    zip_le = LabelEncoder()
    users['ZipPrefix'] = zip_le.fit_transform(users['ZipPrefix'])
    users.drop('Zip-code', axis=1, inplace=True)
    
    # ---------- Merge All ----------
    merged = ratings.merge(users, on="UserID").merge(movies, on="MovieID")
    print("\n合并后的数据前5行:")
    print(merged.head())
    print(f"\n合并后的数据形状: {merged.shape}")
    
    # ---------- Save Processed Data ----------
    output_file = "processed_movielens_data.csv"
    merged.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n数据已成功保存到: {output_file}")
    print(f"保存的文件大小: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")
    
except Exception as e:
    print(f"错误: {e}")