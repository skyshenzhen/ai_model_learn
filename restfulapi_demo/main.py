# -*- coding: utf-8 -*-
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# -------------------------------------------------------------------------------
# Name:         main
# Description:  restfulapi_demo
# Author:       shaver
# Date:         2025/5/20
# -------------------------------------------------------------------------------
# 数据库配置
SQLALCHEMY_DATABASE_URI = 'sqlite:///./students.db'
engine = create_engine(SQLALCHEMY_DATABASE_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# 学生模型
class Student(Base):
    __tablename__ = 'students'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    grade = Column(String, nullable=False)


# 创建数据库表
Base.metadata.create_all(bind=engine)

# FastAPI应用
app = FastAPI()


# Pydantic模型
class StudentCreate(BaseModel):
    name: str
    age: int
    grade: str


# 依赖注入
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 获取所有学生
@app.get("/students/")
def read_students(db=Depends(get_db)):
    students = db.query(Student).all()
    return students


# 创建学生
@app.post("/students/")
def create_student(student: StudentCreate, db=Depends(get_db)):
    db_student = Student(name=student.name, age=student.age, grade=student.grade)
    db.add(db_student)
    db.commit()
    db.refresh(db_student)
    return db_student


# 更新学生
@app.put("/students/{id}")
def update_student(id: int, student: StudentCreate, db=Depends(get_db)):
    db_student = db.query(Student).filter(Student.id == id).first()
    if db_student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    db_student.name = student.name
    db_student.age = student.age
    db_student.grade = student.grade
    db.commit()
    return db_student


# 删除学生
@app.delete("/students/{id}")
def delete_student(id: int, db=Depends(get_db)):
    db_student = db.query(Student).filter(Student.id == id).first()
    if db_student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    db.delete(db_student)
    db.commit()
    return {"message": "Student deleted"}


if __name__ == '__main__':
    import uvicorn
    # 登录地址为 http://127.0.0.1:8081/docs
    uvicorn.run(app, host='127.0.0.1', port=8093)
