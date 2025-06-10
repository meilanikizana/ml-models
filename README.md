# Machine Learning Repository🤖🧠
**BrevityAI** adalah aplikasi berbasis website untuk peringkasan teks dan <i>generate</i> pertanyaan dari teks. Terdapat 2 model machine learning yang digunakan dalam BrevityAI yaitu **model TensorFlow untuk <i>text summarization</i>** dan **model berbasis T5 dari Hugging Face untuk <i>question generation</i>**.

Repository ini terdiri dari kedua model tersebut yang dipisahkan dalam 2 folder sebagai berikut:
```
ml-models/
├──qg-model/
│ ├── Question_Generator_Capstone.ipynb
│ ├── question_generator_capstone.py
├──ts-model/
│ ├── TEXT-SUMMARIZATION-FIX.py
│ ├── TEXT_SUMMARIZATION_CLEAN.ipynb
└──README.md
```

## 📑 Text Summarization Model 📑
Model text summarization merupakan model yang dapat menghasilkan ringkasan dari teks panjang yang diberikan oleh pengguna dalam Bahasa Indonesia. Model ini dibangun dengan pendekatan ekstraktif menggunakan arsitektur Bidirectional LSTM pada framework TensorFlow/Keras yang dilatih pada dataset IndoSum untuk memilih kalimat-kalimat penting dari teks asli.

**1. Dataset**
- Original (Kaggle): [IndoSum-dataset](https://www.kaggle.com/datasets/linkgish/indosum)

**2. Tokenisasi**

    ```
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    input_tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    input_tokenizer.fit_on_texts(X_train)
    ```
Proses tokenisasi dilakukan dengan Keras Tokenizer dari TensorFlow dengan parameter num_words=10000 dan token khusus <OOV> untuk menangani kata-kata yang tidak dikenal. Model menggunakan representasi vektor dari teks yang dipadding hingga panjang 200 token untuk menjaga konsistensi input.

**3. Model TensorFlow**

    ```
    pythonfrom tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, GlobalAveragePooling1D, Dense, Dropout
    model = TextSummarizerModel(
        input_vocab_size=len(input_tokenizer.word_index) + 1,
        output_vocab_size=len(output_tokenizer.word_index) + 1,
        embedding_dim=128
    ).build_model()
    ```
Model menggunakan arsitektur neural network berbasis Bidirectional LSTM dari TensorFlow/Keras untuk memprediksi tingkat kepentingan kalimat. Arsitektur ini memungkinkan model untuk memahami konteks kalimat dari kedua arah dan menentukan kalimat mana yang paling relevan untuk dimasukkan dalam ringkasan.
Model menggunakan arsitektur neural network dengan komponen utama:

- Embedding Layer: Mengubah token menjadi vektor representasi dengan dimensi 128
- Bidirectional LSTM: Memahami konteks kalimat dari dua arah
- Global Average Pooling: Menggabungkan representasi token menjadi representasi kalimat
- Dense Layers dengan Dropout: Mempelajari pola dan mencegah overfitting
- Output Layer: Memprediksi skor kepentingan kalimat (0-1)

**4. Training**

    ```
    history = model.fit(
        X_train_seq, y_train_importance,
        validation_data=(X_val_seq, y_val_importance),
        epochs=10, 
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
    )
    ```
Training dilakukan selama maksimal 10 epoch dengan batch size 32 dan early stopping untuk menghentikan training jika tidak ada peningkatan pada validation loss. Model dilatih untuk memprediksi tingkat kepentingan kalimat berdasarkan konteks dalam teks.

**5. Pemrosesan Teks**

    ```
    def clean_text(text):
        """Pembersihan teks untuk Bahasa Indonesia"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\-()]', '', text)
        return text.strip()
    ```
    ```
    def simple_sentence_tokenize(text):
        """Tokenisasi kalimat sederhana"""
        return re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ').strip())
    ```
Untuk mendukung proses summarization, digunakan fungsi pemrosesan teks khusus Bahasa Indonesia yang membersihkan teks dan memisahkan kalimat berdasarkan tanda baca. Pendekatan ini memungkinkan model untuk memproses teks secara efektif bahkan tanpa library NLP khusus.

### Hugging Face Model
Model ini dapat diakses melalui Hugging Face: [Indonesian-TS-Model](https://huggingface.co/fransiskaarthaa/summarizereal-JS)

## ❔ Question Generation Model ❔
Model question generation (QG) merupakan model yang dapat menghasilkan pertanyaan relevan dari teks yang diberikan oleh pengguna dalam Bahasa Indonesia. Model ini dibangun dengan menggunakan model Transformer berbasis T5 yang kemudian di fine tune dengan dataset SQuAD versi Indonesia.

**1. Dataset**
   - Original (Kaggle) : [squad-variated-indo](https://www.kaggle.com/datasets/mintupsidup/squad-variated-indo)
   - Cleaned : [SQuAD versi Indonesia](https://drive.google.com/file/d/1rAdHLIQJBijlcugcWhAWj_NgP-qe1raW/view?usp=sharing)
     
**2. Tokenisasi**
   ```
   tokenizer = T5Tokenizer.from_pretrained("cahya/t5-base-indonesian-summarization-cased")
   ```
   Proses tokenisasi dilakukan dengan tokenizer T5 yang sesuai dengan model yang akan digunakan yaitu **cahya/t5-base-indonesian-summarization-cased**.
   
**3. Model**
   ```
   t5_model = T5ForConditionalGeneration.from_pretrained("cahya/t5-base-indonesian-summarization-cased")
   ```
   Model pre-trained yang digunakan adalah **[cahya/t5-base-indonesian-summarization-cased](https://huggingface.co/cahya/t5-base-indonesian-summarization-cased)**, model ini merupakan model berbasis T5 versi Bahasa Indonesia yang tersedia di Hugging Face. Model ini merupakan model untuk peringkasan teks dengan basis T5 yang telah di fine tune dalam Bahasa Indonesia.

**4. Fine Tuning**

Fine tuning dilakukan terhadap model pre-trained agar model dapat mempelajari pola antara konteks, pertanyaan dan jawaban sehingga dapat menghasilkan pertanyaan yang relevan dengan konteks sesuai dengan jawabannya. Model pre-trained dilakukan fine tuning terhadap 1000 data dari dataset SQuAD versi Bahasa Indonesia yang sebelumnya telah dilakukan tokenisasi. Fine tuning dilakukan selama 1 epoch dengan batch size 4 dan learning rate 2e-4.

**5. Named Entity Recognition (NER)**

Untuk mendukung proses generasi pertanyaan yang lebih relevan dengan jawaban yang masuk akal ketika proses inferensi, maka digunakan model NER Bahasa Indonesia dari **cahya/bert-base-indonesian-NER**. Model NER ini digunakan untuk mengidentifikasi entitas penting dalam konteks yaitu 'PER' untuk nama orang, 'LOC' untuk lokasi, dan 'DAT' untuk tanggal. Entitas ini kemudian akan diekstrak sebagai jawaban untuk kemudian model membuat pertanyaan yang relevan berdasarkan konteks dan jawaban yang telah diekstrak.

### Hugging Face Model
Model ini dapat diakses melalui Hugging Face: [Indonesian-QG-Model](https://huggingface.co/meilanikizana/indonesia-question-generation-model)

Cara menggunakan model:
```
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("meilanikizana/indonesia-question-generation-model")
model = T5ForConditionalGeneration.from_pretrained("meilanikizana/indonesia-question-generation-model")
```
