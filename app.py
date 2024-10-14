import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import openai
import streamlit as st
import os

def load_course_embeddings():
    if os.path.exists('course_embeddings.npy'):
        return np.load('course_embeddings.npy')
    else:
        st.write("Course embeddings not found.")
        return []

def load_course_data():
    if os.path.exists('course_data.csv'):
        return pd.read_csv('course_data.csv')
    else:
        st.write("Course data CSV not found.")
        return pd.DataFrame()  

openai.api_key = os.getenv("API")

def get_embedding(text):
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        st.write(f"Error while getting embedding: {e}")
        return None

def search_courses(query, threshold=0.8):
    query_embedding = get_embedding(query)
    
    if query_embedding is None:
        return []

    course_embeddings = load_course_embeddings()
    course_data = load_course_data()

    if not len(course_embeddings) or course_data.empty:
        return []

    similarities = cosine_similarity([query_embedding], course_embeddings)[0]

    sorted_indices = np.argsort(similarities)[::-1]

    top_courses = []
    for idx in sorted_indices:
        if similarities[idx] >= threshold: 
            top_courses.append(course_data.iloc[idx]['Course Title'])
            if len(top_courses) == 5:  
                break

    return top_courses

st.write("Enter your query:")
user_query = st.text_input("Search for courses:")

if user_query:
    threshold = 0.75
    results = search_courses(user_query, threshold=threshold)
    
    if results:
        st.write("Top relevant courses:")
        for course_title in results:
            st.write(course_title) 
    else:
        st.write("No courses found with a satisfactory similarity score.")
