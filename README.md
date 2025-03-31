# 电影推荐系统 

### 仅学习使用，如需数据集请联系huanghy0820@163.com

## 项目概述

这是一个基于Neo4j图数据库的智能电影推荐系统，采用协同过滤算法和内容过滤相结合的方式，为用户提供个性化电影推荐服务。系统通过分析用户历史评分、相似用户行为以及电影属性特征，计算出最适合推荐给目标用户的电影列表。

## 系统架构

```
数据层(CSV文件) → 数据加载层(Neo4j) → 推荐算法层 → 用户交互层
```

## 功能特性

1. **核心功能**
   - 个性化电影推荐
   - 用户评分历史查询
   - 电影相似度计算

2. **高级功能**
   - 基于类型的过滤系统
   - 可调节的推荐参数
   - 数据重新加载功能

3. **算法特性**
   - 余弦相似度计算用户相似性
   - 加权平均预测评分
   - 多维度排序(评分+推荐人数)

## 系统依赖

### 必需组件
- Neo4j 4.x+ 图数据库
- Python 3.8+
- Neo4j Python Driver
- pandas 数据分析库

### 数据文件
- out_movies.csv
- out_grades.csv  
- out_genre.csv
- out_keywords.csv
- out_productor.csv

## 安装与配置

1. **数据库配置**
```bash
# 修改连接配置
uri = "bolt://localhost:7687"
auth=("neo4j", "your_password")
```

2. **数据准备**
- 将所有CSV文件放入Neo4j的import目录
- 确保文件权限正确

3. **参数调整**
```python
# 推荐系统参数
k = 10                   # 考虑的相似用户数
movies_common = 3        # 最小共同评分电影数  
user_common = 2          # 最小推荐用户数
threshold_sim = 0.9      # 相似度阈值
```

## 使用指南

### 1. 系统初始化
```
是否需要重新加载并创建知识图谱？(输入0或1)
1
初始化完成...
```

### 2. 主交互流程
```
请输入要为哪位用户推荐电影，输入其ID即可（回车结束）：
101
为该用户推荐多少个电影呢？
5
是否需要过滤不喜欢的类型？（输入0或1）
1

               genre
0              Drama
1             Comedy
2           Thriller
...
输入不喜欢的类型索引即可，例如：1 2 3
2 5
```

### 3. 输出示例
```
Your ratings are the following:

title                    grade
"The Shawshank Redemption" 5
"The Godfather"           4
...

Recommended movies:

title                    avg grade  num recommenders  genres
"Pulp Fiction"           4.8       12                ["Crime", "Drama"]
"Inception"              4.7       10                ["Action", "Sci-Fi"]
...
```

## 算法详解

### 1. 数据加载模型
```cypher
LOAD CSV WITH HEADERS FROM "file:///out_movies.csv" 
CREATE (:MOVIE {title: csv.title})
```

### 2. 用户相似度计算
```python
SUM(r1.grading * r2.grading)/(SQRT(SUM(r1.grading^2)) * SQRT(SUM(r2.grading^2))) AS sim
```

### 3. 推荐生成算法
```cypher
WITH
    m.title AS title,
    SUM(r.grading * s.sim)/SUM(s.sim) AS grade,  # 加权平均分
    COUNT(u2) AS num,                            # 推荐人数
    gen
WHERE num >= {user_common}
ORDER BY grade DESC, num DESC                    # 双重排序
```

## 高级配置

### 性能调优参数
| 参数          | 说明               | 推荐值   |
| ------------- | ------------------ | -------- |
| k             | 相似用户数量       | 5-20     |
| movies_common | 最小共同评分电影数 | 3-5      |
| threshold_sim | 相似度阈值         | 0.8-0.95 |
| user_common   | 最小推荐用户数     | 2-5      |

### 类型过滤语法
```python
Q_GENRE = "AND (ANY(x IN "+str(genres)+" WHERE x in gen)"
```

## 常见问题

1. **数据加载失败**
   - 检查CSV文件路径
   - 验证Neo4j import目录权限
   - 确认文件编码为UTF-8

2. **推荐结果不理想**  
   - 调整相似度阈值
   - 增加movies_common值
   - 检查原始评分数据质量

3. **性能问题**
   - 为常用查询创建索引
   - 减少k值
   - 优化Cypher查询

## 扩展开发

1. **添加新数据源**
```python
LOAD CSV WITH HEADERS FROM "file:///new_data.csv" AS csv
MERGE (m:Movie {title: csv.title})
MERGE (n:NewNode {prop: csv.value})
CREATE (m)-[:NEW_RELATIONSHIP]->(n)
```

2. **实现实时更新**
```python
def update_rating(user_id, movie_title, grade):
    with driver.session() as session:
        session.run("""
            MERGE (u:User {id: $user_id})
            MERGE (m:Movie {title: $movie_title}) 
            MERGE (u)-[r:RATED]->(m)
            SET r.grading = $grade
            """, 
            parameters={"user_id": user_id, "movie_title": movie_title, "grade": grade})
```
