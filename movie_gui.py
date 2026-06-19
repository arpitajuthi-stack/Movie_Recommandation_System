import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================
# Load Dataset
# ==========================
movies = pd.read_csv("movies.csv")

vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies["genre"])

similarity = cosine_similarity(genre_matrix)


# ==========================
# Recommendation Function
# ==========================
def recommend_movies():

    movie_name = movie_entry.get().strip()

    if movie_name == "":
        messagebox.showwarning(
            "Warning",
            "Please enter a movie name."
        )
        return

    result_box.delete(0, tk.END)

    try:

        movie_index = movies[
            movies["title"].str.lower()
            == movie_name.lower()
        ].index[0]

        scores = list(
            enumerate(similarity[movie_index])
        )

        scores = sorted(
            scores,
            key=lambda x: x[1],
            reverse=True
        )

        for movie in scores[1:6]:

            result_box.insert(
                tk.END,
                movies.iloc[movie[0]]["title"]
            )

    except:
        messagebox.showerror(
            "Error",
            "Movie not found!"
        )


# ==========================
# GUI Window
# ==========================
root = tk.Tk()

root.title("Movie Recommendation System")
root.geometry("700x500")
root.resizable(False, False)

# Heading
title_label = tk.Label(
    root,
    text="🎬 Movie Recommendation System",
    font=("Arial", 20, "bold")
)

title_label.pack(pady=20)

# Input Frame
input_frame = tk.Frame(root)

input_frame.pack(pady=10)

movie_label = tk.Label(
    input_frame,
    text="Movie Name:",
    font=("Arial", 12)
)

movie_label.grid(row=0, column=0, padx=10)

movie_entry = ttk.Entry(
    input_frame,
    width=30
)

movie_entry.grid(row=0, column=1)

# Button
recommend_button = ttk.Button(
    root,
    text="Recommend Movies",
    command=recommend_movies
)

recommend_button.pack(pady=15)

# Result Label
result_label = tk.Label(
    root,
    text="Recommended Movies",
    font=("Arial", 14, "bold")
)

result_label.pack()

# Result Box
result_box = tk.Listbox(
    root,
    width=50,
    height=10,
    font=("Arial", 12)
)

result_box.pack(pady=10)

# Footer
footer = tk.Label(
    root,
    text="Developed using Python, Pandas & Machine Learning",
    font=("Arial", 10)
)

footer.pack(side=tk.BOTTOM, pady=10)

root.mainloop()