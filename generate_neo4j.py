from neo4j import GraphDatabase
import pandas as pd

# 连接Neo4j数据库
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "@Xh19990820"))

# 相关设置和参数
k = 10
movies_common = 3
user_common = 2
threshold_sim = 0.9

def load_data():
    with driver.session() as session:
        session.run("""MATCH ()-[r]->() DELETE r""")
        session.run("""MATCH (r) DELETE r""")

        print("Loading movies...")
        session.run("""
            LOAD CSV WITH HEADERS FROM "file:///out_movies.csv" AS csv
            CREATE (:MOVIE {title: csv.title})
        """)

        print("Loading gradings...")
        session.run("""
            LOAD CSV WITH HEADERS FROM "file:///out_grades.csv" AS csv
            MERGE (m:Movie {title: csv.title})
            MERGE (u:User {id: toInteger(csv.user_id)})
            CREATE (u)-[:RATED {grading : toInteger(csv.grade)}]->(m)
        """)

        print("Loading genres...")
        session.run("""
            LOAD CSV WITH HEADERS FROM "file:///out_genre.csv" AS csv
            MERGE (m:Movie {title: csv.title})
            MERGE (g:Genre {genre: csv.genre})
            CREATE (m)-[:HSD_GENRE]->(g)
        """)

        print("Loading keywords...")
        session.run("""
            LOAD CSV WITH HEADERS FROM "file:///out_keywords.csv" AS csv
            MERGE (m:Movie {title: csv.title})
            MERGE (k:Keyword {keyword: csv.keyword})
            CREATE (m)-[:HSD_KEYWORD]->(k)
        """)

        print("Loading productors...")
        session.run("""
            LOAD CSV WITH HEADERS FROM "file:///out_productor.csv" AS csv
            MERGE (m:Movie {title: csv.title})
            MERGE (p:Productor {name: csv.productor})
            CREATE (m)-[:HSD_PRODUCTOR]->(p)
        """)
    driver.close()

def queries():
    while True:
        userid = input("请输入要为哪位用户推荐电影，输入其ID即可（回车结束）：\n")

        if userid == "":
            break

        userid = int(userid)
        m = int(input("为该用户推荐多少个电影呢？\n"))

        genres = []
        if int(input("是否需要过滤不喜欢的类型？（输入0或1）\n")):
            with driver.session() as session:
                try:
                    q = session.run(f"""MATCH (g:Genre) RETURN g.genre AS genre""")
                    result = []
                    for i, r in enumerate(q):
                        result.append(r["genre"])
                    df = pd.DataFrame(result, columns={"genre"})
                    print()
                    print(df)

                    inp = input("输入不喜欢的类型索引即可，例如：1 2 3")
                    if len(inp) != 0:
                        inp = inp.split(" ")
                        genres = [df["genre"].iloc[int(x)] for x in inp]
                except:
                    print("Error")

        # 找到当前ID
        with driver.session() as session:
            q = session.run(f"""
                    MATCH (u1:User {{id : {userid}}})-[r:RATED]-(m:Movie)
                    RETURN m.title AS title, r.grading AS grade
                    ORDER BY grade DESC
                """)

            print()
            print("Your ratings are the following（你的评分如下）:")

            result = []
            for r in q:
                result.append([r["title"], r["grade"]])

            if len(result) == 0:
                print("No ratings found")
            else:
                df = pd.DataFrame(result, columns=["title", "grade"])
                print()
                print(df.to_string(index=False))
            print()

            session.run(f"""
                MATCH (u1:User)-[s:SIMILARITY]-(u2:User)
                DELETE s
            """)

            # 找到当前用户评分的电影以及这些电影被其他用户评分的用户，with是吧查询集合当做结果方便后面用where余弦相似度计算
            """
            Cosine相似度计算法(Cosine Similarity)
            """
            session.run(f"""
                    MATCH (u1:User {{id : {userid}}})-[r1:RATED]-(m:Movie)-[r2:RATED]-(u2:User)
                    WITH
                        u1, u2,
                        COUNT(m) AS movies_common,
                        SUM(r1.grading * r2.grading)/(SQRT(SUM(r1.grading^2)) * SQRT(SUM(r2.grading^2))) AS sim
                    WHERE movies_common >= {movies_common} AND sim > {threshold_sim}
                    MERGE (u1)-[s:SIMILARITY]-(u2)
                    SET s.sim = sim
            """)

            # 过滤操作
            Q_GENRE = ""
            if (len(genres) > 0):
                Q_GENRE = "AND ((SIZE(gen) > 0) AND a"
                Q_GENRE += "(ANY(x IN " + str(genres) + " WHERE x in gen))"
                Q_GENRE += ")"

            # 找到相似的用户，然后看他们喜欢什么电影Collect，将所有值收集到一个集合List中
            """
            s:SIMILARITY通过关系边查询
            ORDER BY 降序排列
            """
            q = session.run(f"""
                    MATCH (u1:User {{id : {userid}}})-[s:SIMILARITY]-(u2:User)
                    WITH u1, u2, s
                    ORDER BY s.sim DESC LIMIT {k}
                    MATCH (m:Movie)-[r:RATED]-(u2)
                    OPTIONAL MATCH (g:Genre)--(m)
                    WITH u1, u2, s, m, r, COLLECT(DISTINCT g.genre) AS gen
                    WHERE NOT((m)-[:RATED]-(u1)) {Q_GENRE}
                    WITH
                        m.title AS title,
                        SUM(r.grading * s.sim)/SUM(s.sim) AS grade,
                        COUNT(u2) AS num,
                        gen
                    WHERE num >= {user_common}
                    RETURN title, grade, num, gen
                    ORDER BY grade DESC, num DESC
                    LIMIT {m}
                """)

            print("Recommended movies（推荐电影如下）:")

            result = []
            for r in q:
                result.append([r["title"], r["grade"], r["num"], r["gen"]])
            if len(result) == 0:
                print("No recommendations found（没有找到适合推荐的）")
                print()
                continue

            df = pd.DataFrame(result, columns=["title", "avg grade", "num recommenders", "genres"])
            print()
            print(df.to_string(index=False))
            print()

        driver.close()

if __name__ == "__main__":
    if int(input("是否需要重新加载并创建知识图谱？(输入0或1)\n")):
        load_data()
        print("初始化完成...")

    queries()
