import pandas as pd

from generate_neo4j import *

def queries():
    while True:
        userid = input("请输入要为哪位用户推荐电影，输入其ID即可:\n")

        if userid == "":
            break

        userid = int(userid)
        m = int(input("为该用户推荐多少个电影呢?\n"))

        genres = []
        if int(input("是否需要过滤不喜欢的类型？(输入0或1)\n")):
            with driver.session() as session:
                try:
                    q = session.run(f"""MATCH (g:Genre) RETURN g.genres AS genre""")
                    result = []
                    for i,r in enumerate(q):
                        result.append(r['genre'])
                    df = pd.DataFrame(result,columns={"genre"})
                    print()
                    print(df)

                    inp = input("输入不喜欢的类型索引即可：")
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
            print("your ratings are the following:")

            result = []
            for r in q:
                result.append([r['title'],r["grade"]])

            if len(result) == 0:
                print("No ratings found")
            else:
                df = pd.DataFrame(result,columns=["title","grade"])
                print()
                print(df.to_string(index=False))
            print()

            session.run(f"""
                MATCH (u1:User)-[s:SIMILARITY]-(u2:User)
                DELETE s
            """)

            """
            Cosine相似度计算法(Cosine Similarity)
            """
            session.run(f"""
                MATCH (u1:User {{id : {userid}}})-[r1:RATED]-(m:Movie)-[r2:RATED]-(u2:User)
                WITH
                    u1,u2,
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

            """
            s:SIMILARITY通过关系边查询
            ORDER BY 降序排列
            """
            q = session.run(f"""
                MATCH (u1:User {{id : {userid}}})-[s:SIMILARITY]-(u2:User)
                WITH u1,u2,s
                ORDER BY s.sim DESC LIMIT {k}
                MATCH (m:Movie)-[r:RATED]-(u2)
                OPTIONAL MATCH (g:Genre)--(m)
                WITH u1,u2,s,m,r,COLLECT(DISTINCT g.genre) AS gen
                WHERE NOT((m)-[:RATED]-(u1)) {Q_GENRE}
                WITH
                    m.title AS title,
                    SUM(r.grading * s.sim)/SUM(s.sim) AS grade,
                    COUNT(u2) AS num,
                    gen
                WHERE num >= {user_common}
                RETURN title,grade,num,gen
                ORDER BY grade DESC,num DESC
                LIMIT {m}
            """)

            print("Recommended movies:")

            result = []
            for r in q:
                result.append([r['title'],r['grade'],r['num'],r['gen']])
            if len(result) == 0:
                print("No recommended movies found")
                print()
                continue

            df = pd.DataFrame(result,columns=["title","avg grade","num recommenders","genres"])
            print()
            print(df.to_string(index=False))
            print()

if __name__ == "__main__":
    queries()