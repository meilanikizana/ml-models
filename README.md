# Machine Learning Repository
**BrevityAI** adalah aplikasi berbasis website untuk peringkasan teks dan <i>generate</i> pertanyaan dari teks. Terdapat 2 model machine learning yang digunakan dalam BrevityAI yaitu **model TensorFlow untuk <i>text summarization</i>** dan **model berbasis T5 dari Hugging Face untuk <i>question generation</i>**.

Repository ini terdiri dari kedua model tersebut yang dipisahkan dalam 2 folder sebagai berikut:
```
ml-models/
├──qg-model/
│ ├── Question_Generator_Capstone.ipynb
│ ├── question_generator_capstone.py
├──ts-model/
│ ├── Text_Summarization_Caps.ipynb
│ ├── text_summarization_capstone.py(1).py
└──README.md
```

## Text Summarization Model

## Question Generation Model
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
