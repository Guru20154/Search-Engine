import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import streamlit as st
import os

course_data = []

for x in range(1, 9):
    url = "https://courses.analyticsvidhya.com/collections/courses?page=" + str(x)

    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        course_cards = soup.find_all('a', class_='course-card course-card__public published')

        for course_card in course_cards:
            price_tag = course_card.find('span', class_='course-card__price')
            if price_tag:
                strong_tag = price_tag.find('strong')
                if strong_tag and 'Free' in strong_tag.get_text(strip=True):
                    course_link = course_card['href']

                    course_response = requests.get("https://courses.analyticsvidhya.com" + course_link)

                    if course_response.status_code == 200:
                        course_soup = BeautifulSoup(course_response.text, 'html.parser')

                        h1_tag = course_soup.find('h1', class_='section__heading')
                        h1_text = h1_tag.get_text(strip=True) if h1_tag else "No Title"

                        p_tag = course_soup.find('section', class_='section__body').find('p')
                        p_text = p_tag.get_text(strip=True) if p_tag else "No Description"

                        chapters = course_soup.find_all('article', class_='section__content')

                        all_chapter_titles = []  
                        all_lessons = []  

                        for chapter in chapters:
                            h5_tags = chapter.find_all('h5', class_='course-curriculum__chapter-title')
                            for h5_tag in h5_tags:
                                chapter_title = h5_tag.get_text(strip=True)
                                all_chapter_titles.append(chapter_title)  

                                ul = h5_tag.find_parent().find_next_sibling('ul')
                                if ul:
                                    li_items = [li.get_text(strip=True) for li in ul.find_all('li')]
                                    lessons_str = ", ".join(li_items)  
                                    all_lessons.append(lessons_str)  

                        all_chapter_titles_str = ", ".join(all_chapter_titles) if all_chapter_titles else "No Chapters"
                        all_lessons_str = ", ".join(all_lessons) if all_lessons else "No Lessons"

                        course_data.append({
                            'Course Title': h1_text,
                            'Description': p_text,
                            'All Chapter Titles': all_chapter_titles_str,
                            'All Lessons': all_lessons_str
                        })

                    else:
                        print(f"Failed to retrieve course page. Status code: {course_response.status_code}")
    else:
        print(f"Failed to retrieve the main page. Status code: {response.status_code}")

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
        print(f"Error while getting embedding: {e}")
        return None

course_embeddings = []
for course in course_data:
    text = f"{course['Course Title']} {course['Description']} {course['All Chapter Titles']} {course['All Lessons']}"
    course_embedding = get_embedding(text)
    course_embeddings.append(course_embedding)

def search_courses(query):
    query_embedding = get_embedding(query)

    similarities = cosine_similarity([query_embedding], course_embeddings)

    sorted_indices = np.argsort(similarities[0])[::-1]

    top_courses = [course_data[idx]['Course Title'] for idx in sorted_indices[:3]]

    return top_courses

st.write("Enter your query:")
user_query = st.text_input("Search for courses:")

if user_query:
    results = search_courses(user_query)
    st.write("Top relevant courses:")
    st.write(results)