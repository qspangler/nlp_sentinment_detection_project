import tkinter as tk
from tkinter import ttk
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.signal import savgol_filter

# Sentiment analyzer and emotion classifier
sia = SentimentIntensityAnalyzer()
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True, framework="pt")

# Analyze sentiment and emotions for each sentence
def analyze_sentiment_and_emotions(text):
    sentences = sent_tokenize(text)
    emotions_per_sentence = []
    sentiment_scores = []
    for sentence in sentences:
        scores = emotion_classifier(sentence)[0]
        emotions_per_sentence.append({score['label']: score['score'] for score in scores})
        sentiment_scores.append(sia.polarity_scores(sentence)['compound'])
    return sentences, emotions_per_sentence, sentiment_scores

# Search for sentences containing the keywords
def search_sentences():
    keywords = entry.get().lower().split()
    matching_sentences = [
        s for s in sentences if all(keyword in s.lower() for keyword in keywords)
    ]
    listbox.delete(0, tk.END)
    for sentence in matching_sentences:
        listbox.insert(tk.END, sentence)

# Display emotion bar chart with subtitle and highlight selected sentence
def show_emotion_bar_chart(emotion_scores, subtitle):
    labels = list(emotion_scores.keys())
    sizes = list(emotion_scores.values())
    
    # Define colors for each emotion
    colors = {
        'anger': '#FF6347',     # Tomato Red
        'joy': '#FFD700',       # Gold Yellow
        'sadness': '#1E90FF',   # Dodger Blue
        'fear': '#9370DB',      # Medium Purple
        'surprise': '#FFA500',  # Orange
        'disgust': '#32CD32',   # Lime Green
        'neutral': '#A9A9A9'    # Dark Gray
    }
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.bar(labels, sizes, color=[colors.get(label, 'gray') for label in labels], edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Scores')
    ax.set_title('Emotion Scores')
    
    # Subtitle with the sentence
    ax.text(0.5, -0.25, subtitle, ha='center', va='center', transform=ax.transAxes, fontsize=10, wrap=True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.subplots_adjust(hspace=0.5)  # Add more vertical space between plots

    # Clear if exists
    if hasattr(show_emotion_bar_chart, "canvas"):
        show_emotion_bar_chart.canvas.get_tk_widget().pack_forget()

    show_emotion_bar_chart.canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    show_emotion_bar_chart.canvas.draw()
    show_emotion_bar_chart.canvas.get_tk_widget().pack()

# Display sentiment line chart with highlighted selected sentence
def show_sentiment_line_chart(sentiment_scores, highlight_index=None):
    smoothed_scores = savgol_filter(sentiment_scores, min(len(sentiment_scores), 200), 3)

    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(sentiment_scores, label='Original Sentiment Scores', color='black')
    ax.plot(smoothed_scores, label='Smoothed Sentiment Scores', color='gold')

    if highlight_index is not None:
        ax.plot(highlight_index, sentiment_scores[highlight_index], 'ro')  # Highlight selected sentence

    ax.set_xlabel('Sentence Index')
    ax.set_ylabel('Sentiment Score')
    ax.set_title('Sentiment Over Movie')
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)

    plt.subplots_adjust(hspace=0.5)  # Add more vertical space between plots

    if hasattr(show_sentiment_line_chart, "canvas"):
        show_sentiment_line_chart.canvas.get_tk_widget().pack_forget()

    show_sentiment_line_chart.canvas = FigureCanvasTkAgg(fig, master=line_chart_frame)
    show_sentiment_line_chart.canvas.draw()
    show_sentiment_line_chart.canvas.get_tk_widget().pack()

# URL to fetch text from Bee Movie script
url = 'https://courses.cs.washington.edu/courses/cse163/20wi/files/lectures/L04/bee-movie.txt'
response = requests.get(url)

if response.status_code == 200:
    text = response.text
else:
    print(f"Failed to retrieve the file: {response.status_code}")

soup = BeautifulSoup(response.content, 'html.parser')
text = soup.get_text()
sentences, emotions_per_sentence, sentiment_scores = analyze_sentiment_and_emotions(text)

# GUI setup
root = tk.Tk()
root.title("Bee Movie Sentiment Analysis Tool")
root.geometry("1200x800")
root.configure(bg='#f0f0f0')

# Hold search and chart areas side by side
main_frame = ttk.Frame(root)
main_frame.pack(pady=20)

# Search area setup
search_frame = ttk.Frame(main_frame)
search_frame.pack(side=tk.LEFT, padx=20)

entry_label = ttk.Label(search_frame, text="Enter a line, phrase, or keyword from the movie:", font=("Helvetica", 12))
entry_label.pack(pady=5)

entry = ttk.Entry(search_frame, width=50, font=("Helvetica", 12))
entry.pack(pady=5)

search_button = ttk.Button(search_frame, text="Search", command=search_sentences)
search_button.pack(pady=5)

listbox_frame = ttk.Frame(search_frame)
listbox_frame.pack(pady=10)

listbox_scrollbar = ttk.Scrollbar(listbox_frame)
listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

listbox = tk.Listbox(listbox_frame, width=50, height=20, yscrollcommand=listbox_scrollbar.set, font=("Helvetica", 10))
listbox.pack()

listbox_scrollbar.config(command=listbox.yview)

# Chart area setup with two frames: one for bar chart and one for line chart
chart_frame = ttk.Frame(main_frame)
chart_frame.pack(side=tk.RIGHT)

line_chart_frame = ttk.Frame(chart_frame)
line_chart_frame.pack(pady=10)

def on_select(event):
    selected_index = listbox.curselection()
    if selected_index:
        selected_sentence = listbox.get(selected_index)
        index = sentences.index(selected_sentence)
        emotion_scores = emotions_per_sentence[index]
        
        show_emotion_bar_chart(emotion_scores, selected_sentence)
        show_sentiment_line_chart(sentiment_scores, index)  # Highlight selected sentence

listbox.bind('<<ListboxSelect>>', on_select)

root.mainloop()