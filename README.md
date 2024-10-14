# Course Recommendation System

This project is a course recommendation system that uses OpenAI embeddings and cosine similarity to recommend courses based on user queries. The system leverages precomputed course embeddings and matches user queries to these embeddings to find the most relevant courses.

## Features

- Retrieves course data and computes embeddings for course descriptions.
- Stores course data and embeddings locally for faster subsequent queries.
- Uses OpenAI's `text-embedding-ada-002` model to generate embeddings.
- Finds courses similar to the user's query based on cosine similarity.
- Displays the top 5 relevant courses that have a similarity score above 0.8.

## Prerequisites

Before running the code, ensure you have the following:

- Python 3.x
- A valid OpenAI API key.
- Required Python packages (see below).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/course-recommendation-system.git
   cd course-recommendation-system
   ```
2. **Install Required Packages**: 
   Create a requirements.txt file if not provided, and install the dependencies:
   ```bash
      pip install -r requirements.txt
   ```
   requirements.txt content:
   ```txt
   numpy
   pandas
   scikit-learn
   openai
   streamlit
   requests
   beautifulsoup4
   ```

3. Add OpenAI API Key: Ensure your OpenAI API key is set as an environment variable:
   ```bash
   export API=<your-openai-api-key>
   ```

## Usage
1. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```
2. **Enter a Query**: When prompted, enter a query describing the type of courses you are looking for.
3. **View Recommended Courses**: The application will display the top 5 courses that match your query with a similarity score above 0.8.

# Course Search Application

This is a course search application that leverages OpenAI's embedding model to find relevant courses based on user queries. It fetches course data and precomputed embeddings to provide quick and accurate search results.

## How It Works

1. **Data Collection**: Initially, the application scrapes course data from the Analytics Vidhya website. It collects course titles, descriptions, chapters, and lessons and stores this information in a CSV file (`course_data.csv`).

2. **Embedding Generation**: The application generates embeddings for each course using OpenAI's `text-embedding-ada-002` model. These embeddings are stored in a NumPy array file (`course_embeddings.npy`), allowing for quick retrieval during searches.

3. **User Query Input**: When a user inputs a query through the Streamlit interface, the application generates an embedding for the query.

4. **Similarity Calculation**: The application computes the cosine similarity between the query embedding and the precomputed course embeddings. Only courses with a similarity score above a threshold (0.8) are considered relevant.

5. **Result Display**: The top 5 courses matching the user's query are displayed as results, focusing on course titles to provide a clean and informative user experience.

## Files

- **`app.py`**: The main code file containing logic for loading embeddings, handling user input, and displaying results.
- **`course_data.csv`**: Stores course titles, descriptions, chapters, and lessons.
- **`course_embeddings.npy`**: Contains the precomputed embeddings for each course. This file is loaded directly to speed up the search process.
- **`README.md`**: Provides an overview of the project, setup instructions, and usage details.

## Notes

- This version assumes that the initial data scraping and embeddings generation has already been completed.
- The `course_data.csv` and `course_embeddings.npy` files must be present in the same directory as `app.py` for the system to function correctly.
- Make sure to replace `<your-openai-api-key>` with your actual API key in the `API` environment variable.
