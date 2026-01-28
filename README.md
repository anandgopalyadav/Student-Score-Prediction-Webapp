# 🎓 Student Score Prediction Web App (Linear Regression + Flask)

A **Student Score Prediction Web Application** built using **Simple Linear Regression** and **Flask**.  
This project predicts a student’s **Score (%)** based on **Hours Studied** and displays the result instantly with a **professional regression graph**.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Flask](https://img.shields.io/badge/Flask-WebApp-black)
![ML](https://img.shields.io/badge/Machine%20Learning-Linear%20Regression-orange)
![License](https://img.shields.io/badge/License-MIT-green)

✅ Score prediction is capped between **0% and 100%**  
✅ Study hours validation: **0 to 20 hours/day**  
✅ Graph includes **dataset points + regression line + predicted point**

---

## 🚀 Features

- Auto-loads dataset from `data/student_scores.csv`
- Data exploration and visualization
- Train/Test split
- Trains **Linear Regression** model
- Predicts student score percentage
- Flask Web App interface for user input
- Input validation:
  - ❌ Hours cannot be negative
  - ❌ Hours cannot exceed **20 hours/day**
  - ✅ Predicted score always stays between **0% and 100%**
- Professional graph in frontend:
  - Scatter plot (Actual dataset)
  - Regression line (Best fit)
  - Predicted point highlighted
- Saves predictions to `outputs/predictions.csv`

---

## 🛠️ Technologies Used

- Python
- Flask
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 📂 Dataset

The dataset must contain **2 columns**:

| Hours | Scores |
|------:|------:|
| 2.5   | 21    |
| 5.1   | 47    |
| 7.8   | 78    |

✅ Dataset file location:
```bash
data/student_scores.csv
```

---

## 📁 Project Structure

```bash
Student-Score-Prediction/
│
├── app.py
├── student_score_prediction.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   └── student_scores.csv
│
├── outputs/
│   └── predictions.csv
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

---

## ✅ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Student-Score-Prediction.git
cd Student-Score-Prediction
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask Web App
```bash
python app.py
```

### 4️⃣ Open in Browser
```bash
http://127.0.0.1:5000/
```

---

## 📊 Output

The web app will display:

✅ Predicted Score (%)  
✅ Graph (Hours vs Score Dataset + Regression Line + Prediction Point)

All predictions are saved automatically in:

```bash
outputs/predictions.csv
```

---

## 📸 Screenshots
<img width="700" height="860" alt="image" src="https://github.com/user-attachments/assets/8c634d16-95a3-4c2b-ae35-c078d5975338" />
<img width="688" height="869" alt="image" src="https://github.com/user-attachments/assets/58303053-1862-423d-8740-806774dfe94b" />

Yello point is the prediction score



Example:
```md
![Home Page](screenshots/home.png)
![Prediction Output](screenshots/result.png)
![Graph Output](screenshots/graph.png)
```

---

## 📈 Machine Learning Model Used

This project uses **Simple Linear Regression**:

**Formula:**
```text
Score = m × Hours + c
```

Where:  
- **m** = slope (coefficient)  
- **c** = intercept  

![Model Graph](image.png)

---

## 👤 Author

**Anand Gopal Yadav**  
📧 Email: [anandgopal2001@gmail.com](mailto:anandgopal2001@gmail.com)  
🔗 LinkedIn: [Anand Gopal Yadav](https://www.linkedin.com/in/anand-gopal-yadav-223964178)  
🐙 GitHub: [anandgopalyadav](https://github.com/anandgopalyadav)

---

## 📜 License

This project is licensed under the **MIT License**.

---

## ⭐ Support

If you found this project useful, please ⭐ the repository!
