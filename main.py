import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import requests
from bs4 import BeautifulSoup
from keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Excel dosyasını yükleme
excel_path = "turkishHeadlines.xlsx"
df = pd.read_excel(excel_path)

# Metin verilerini ve etiketleri ayırma
X = df['HABERLER']
y = df['ETIKET']

# Etiketleri sayısallaştırma
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Metin verilerini vektörize etme
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=200)

# Modeli oluşturma
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=32, input_length=200))
model.add(LSTM(100))
model.add(Dense(7, activation='softmax'))

# Modeli derleme
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Veri setini eğitme ve test verisi ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğitme
model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test))

def get_titles(url, class_name):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        titles = soup.find_all('h3', class_=class_name)
        return [title.text.strip() for title in titles]
    else:
        return []

# Her bir URL ve sınıf
url_class_pairs = [("https://www.mynet.com/haber", "card-text-title h-title py-2 px-3 mb-0"),
                   ("https://www.cnnturk.com/son-dakika-haberleri/", "card-title"),
                   ("https://www.ntv.com.tr/son-dakika", "text-elipsis-3")]

all_titles = []
for url, class_name in url_class_pairs:
    titles = get_titles(url, class_name)
    all_titles.extend(titles)

# Haber başlıklarını vektör formuna dönüştürme
news_sequences = tokenizer.texts_to_sequences(all_titles)
news_sequences = pad_sequences(news_sequences, maxlen=200)

# Haber başlıklarını model ile sınıflandırma
predictions = model.predict(news_sequences)
predicted_categories = np.argmax(predictions, axis=1)

# Sayısal değerleri orijinal sınıf etiketlerine dönüştürme
predicted_categories = label_encoder.inverse_transform(predicted_categories)

# Sınıflandırılmış haberleri bir veri çerçevesine ekleme
df_results = pd.DataFrame({'Haberler': all_titles, 'Tahmin Edilen Kategoriler': predicted_categories})

# Veriyi CSV dosyasına yazdırma
df_results.to_csv('sonuclar.csv', index=False)

# Modeli görselleştirme
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)



