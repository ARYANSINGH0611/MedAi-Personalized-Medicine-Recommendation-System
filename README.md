# Medicine Recommendation System

A machine learning-powered web application for personalized disease prediction and medicine recommendations based on user symptoms.

---

## 1. Dataset Details

- **Datasets Used:**
  - `datasets/Training.csv` — Main ML dataset (symptom-disease mapping)
  - `datasets/description.csv` — Disease descriptions
  - `datasets/medications.csv` — Recommended medications per disease
  - `datasets/precautions_df.csv` — Precautionary measures per disease
  - `datasets/diets.csv` — Dietary recommendations
  - `datasets/workout_df.csv` — Exercise recommendations
  - `datasets/Symptom-severity.csv` — Symptom severity weights
  - `datasets/symtoms_df.csv` — Symptom definitions
  - `datasets/doctors.csv` — Doctor specialty mappings

- **Features:**
  - 132 binary symptom features (0/1 encoding)
  - Target variable: `prognosis` (disease class)
  - Data type: Binary (0/1)

- **Example Features:**
  - `itching` — Skin itching indicator
  - `skin_rash` — Skin rash symptom
  - `high_fever` — Elevated body temperature
  - `cough` — Respiratory cough symptom
  - `vomiting` — Nausea/vomiting indicator
  - `fatigue` — General tiredness
  - `headache` — Head pain symptom
  - `abdominal_pain` — Stomach pain
  - `chest_pain` — Thoracic pain
  - `dizziness` — Balance/vertigo symptom

---

## 2. Data Preprocessing

- Missing values imputed with 0
- Duplicate rows removed
- Label encoding for target (`prognosis`)
- Train-test split: 80% train, 20% test (stratified)
- 5-fold stratified cross-validation

---

## 3. Machine Learning Models

- **Random Forest** — Primary model (used when ≥4 symptoms)
- **SVM (Linear)** — Backup model (used when <4 symptoms)
- Final/best model: Random Forest (200 estimators, balanced class weights)

---

## 4. Performance Metrics

- Cross-validation Accuracy: ~99%
- Test Accuracy: ~99%
- F1 Score (Macro): ~0.99

---

## 5. System Modules

- **Frontend:** Flask-rendered HTML/CSS UI
- **Backend:** Flask REST API (`/predict`, `/details`, `/symptoms`)
- **ML Pipeline:** Pickle-serialized Random Forest + SVM models
- **Data Loader:** CSV loading & O(1) lookup dictionaries
- **Utils:** Symptom parsing, fuzzy matching, confidence calibration

---

## 6. Project Flow

```mermaid
graph TD
    A[User Input (symptoms)] --> B[Parse & Normalize]
    B --> C[Build Feature Vector]
    C --> D[Model Selection (RF/SVM)]
    D --> E[Prediction]
    E --> F[Confidence Calibration]
    F --> G[Output: Disease, Description, Medications, Precautions, Diets, Workouts]
```

---

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```bash
   python train.py
   ```

3. **Run the web app:**
   ```bash
   python app.py
   ```

4. **Open in browser:**  
   Visit [http://localhost:5000](http://localhost:5000)

---

## File Structure

```
app.py
train.py
predict.py
data_loader.py
utils.py
datasets/
templates/
```

---

## License

This project is for academic and research purposes.
