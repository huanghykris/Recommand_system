import pandas as pd
import json
import re

def Netflix():
    print("Loading Netflix-data...")
    print()

    MAX_USER = 1000 # 读取一千个用户
    d_movie = dict() # 字典集
    s_movie = set() # 集合(无序不重复)

    # -----------生成out_movies.csv
    out_movies = open("handle_data/out_movies.csv","w")
    out_movies.write("title\n")

    for line in open("data/movie_titles.csv","r",encoding='ISO-8859-1'):
        line = line.strip().split(',') # 通过","字符进行切割
        movie_id = int(line[0])
        title = line[2].replace("\"","") # 将\字符去掉
        title = "\""+title+"\""

        d_movie[movie_id] = title

        if title in s_movie:
            continue
        s_movie.add(title)

        out_movies.write(f"{title}\n") # 读入数据

    print("out_movies.csv Create Success...")
    out_movies.close()

    # ------------生成out_grade.csv
    out_grade = open("handle_data/out_grades.csv","w")
    out_grade.write("user_id,title,grade\n")

    files = ["data/combined_data_1.txt"]
    for f in files:
        movie_id = -1
        for line in open(f,"r"):
            pos = line.find(":")
            if pos != -1:
                movie_id = int(line[:pos])
                continue
            line = line.strip().split(",")
            user_id = int(line[0])  # 用户编号
            rating = int(line[1])   # 评分

            if user_id > MAX_USER: # 获取1000个用户
                continue

            out_grade.write(f"{user_id},{d_movie[movie_id]},{rating}\n")

    print("out_grade.csv Create Success...")
    out_grade.close()

    # --------------------------
"""
genre.csv数据集 电影类型
keyword.csv数据集 电影关键词
productor.csv数据集 电影导演以及公司
"""
def TMDB():
    print("Loading TMDB data...")
    print()

    pattern = re.compile("[A-Za-z0-9]+")
    out_genre = open("handle_data/out_genre.csv","w",encoding="utf-8")
    out_genre.write("title,genre\n")
    out_keyword = open("handle_data/out_keywords.csv","w",encoding="utf-8")
    out_keyword.write("title,keyword\n")
    out_productor = open("handle_data/out_productor.csv","w",encoding="utf-8")
    out_productor.write("title,productor\n")

    df = pd.read_csv("data/tmdb_5000_movies.csv",sep=",")
    json_columns = ['genres','keywords','production_companies']
    for column in json_columns:
        df[column] = df[column].apply(json.loads) # 处理字典
    df = df[["genres","keywords","original_title","production_companies"]]

    for _, row in df.iterrows():
        title = row["original_title"]
        if not pattern.fullmatch(title):
            continue
        title = "\""+title+"\""

        for g in row["genres"]:
            genre = g["name"]
            genre = "\""+genre+"\""
            out_genre.write(f"{title},{genre}\n")

        for g in row["keywords"]:
            keyword = g["name"]
            keyword = "\""+keyword+"\""
            out_keyword.write(f"{title},{keyword}\n")

        for g in row["production_companies"]:
            production_company = g["name"]
            production_company = "\""+production_company+"\""
            out_productor.write(f"{title},{production_company}\n")

    print("out_genre.csv Create Success...")
    print("out_keyword.csv Create Success...")
    print("out_productor.csv Create Success...")
    out_genre.close()
    out_keyword.close()
    out_productor.close()

if __name__ == "__main__":
    Netflix()
    print("="*40)
    TMDB()