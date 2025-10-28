# 🎬 Content-Based Movie Recommender System

A **Content-Based Recommender System** that suggests similar movies based on textual features like *overview, genres, keywords, cast,* and *crew*.  
This project uses **Natural Language Processing (NLP)** and **Cosine Similarity** to recommend movies similar to the one a user selects.

---

## 🚀 Features

- ✅ Combines movie metadata from multiple datasets  
- ✅ Extracts and cleans text-based features (`overview`, `genres`, `keywords`, `cast`, `crew`)  
- ✅ Applies **Stemming** and **Vectorization (CountVectorizer)**  
- ✅ Calculates **Cosine Similarity** between movies  
- ✅ Recommends top 5 similar movies  
- ✅ Saves the model artifacts (`movie_dict.pkl` and `similarity.pkl`) for deployment  

---

## 🧠 Tech Stack

- **Python 3.x**
- **Libraries:**
  - `numpy`
  - `pandas`
  - `nltk`
  - `scikit-learn`
  - `pickle`
- **Dataset:** [TMDB 5000 Movies Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Environment:** Google Colab

---

## 📁 Dataset

You’ll need to place these files in your Google Drive under:  
```
drive/My Drive/Datasets/
```

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

---

## ⚙️ Installation & Setup

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/Content-Based-Recommender-System.git
cd Content-Based-Recommender-System
```

### 2. Open in Google Colab
Upload the notebook and mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Install dependencies
```bash
pip install numpy pandas scikit-learn nltk
```

---

## 🧩 Data Preprocessing Steps

1. **Merge Datasets:** Combine `movies` and `credits` on the title.  
2. **Select Features:** Keep important columns —  
   `movie_id, title, overview, genres, keywords, cast, crew, release_date, vote_average`.  
3. **Handle Missing & Duplicate Values.**  
4. **Parse JSON Columns:** Convert stringified lists (`genres`, `keywords`, `cast`, `crew`) using `ast.literal_eval()`.  
5. **Feature Engineering:**  
   - Extract top 3 cast members.  
   - Identify the director.  
   - Combine all textual data into a single column called `tags`.  
6. **Text Cleaning:** Remove spaces and convert to lowercase.  
7. **Stemming:** Use `nltk.PorterStemmer` to reduce words to their base form.  
8. **Vectorization:** Convert text to numerical vectors using `CountVectorizer`.  
9. **Similarity Matrix:** Compute **Cosine Similarity** between all movie vectors.

---

## 🎯 Recommendation Function

```python
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),
                       reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)
```

**Example:**
```python
recommend('Spider-Man 2')
```

**Output:**
```
Spider-Man
The Amazing Spider-Man
Spider-Man 3
The Amazing Spider-Man 2
Avengers: Age of Ultron
```

---

## 💾 Model Saving

To save the processed data and similarity matrix:
```python
import pickle, os

save_path = '/content/drive/My Drive/artifacts'
os.makedirs(save_path, exist_ok=True)

pickle.dump(new_df.to_dict(), open(os.path.join(save_path, 'movie_dict.pkl'), 'wb'))
pickle.dump(similarity, open(os.path.join(save_path, 'similarity.pkl'), 'wb'))
```

---

## 📊 Future Improvements

- Add a **Streamlit** or **Flask** web app interface  
- Use **TF-IDF Vectorizer** for better text weighting  
- Integrate **user-based recommendations** for hybrid systems  
- Deploy using **Render** or **Hugging Face Spaces**

---

## Author
**Udit Kaushik**  
uditkaushikk555@gmail.com  
[LinkedIn](https://www.linkedin.com/in/udit-kaushik-883341377)  


