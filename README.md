# 🎓 Student Score Prediction Web App (Linear Regression + Flask)

This project predicts a student's **Score (%)** based on **Hours Studied** using **Simple Linear Regression** in Python.  
It also provides a **Flask-based Web App** where users can enter study hours and instantly get the predicted score along with a **professional graph**.

✅ Prediction is capped between **0% and 100%**  
✅ Study hours validation: **0 to 20 hours only**  
✅ Graph shows **dataset + regression line + predicted point**

---

## 🚀 Features
- ✅ Auto loads dataset from `data/student_scores.csv`
- ✅ Data exploration & visualization
- ✅ Train/Test split
- ✅ Train Linear Regression model
- ✅ Predict Score Percentage
- ✅ Flask Web App frontend input
- ✅ Validation:
  - ❌ Hours cannot be negative
  - ❌ Hours cannot be more than 20 hours/day
  - ✅ Predicted score always between 0 and 100
- ✅ Professional graph in frontend:
  - Scatter plot (Actual dataset)
  - Regression line (Best fit)
  - Predicted point highlight
- ✅ Saves predictions to `outputs/predictions.csv`

---

## 🛠️ Technologies Used
- Python
- Flask
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 📂 Dataset Format
Your dataset must contain 2 columns:

| Hours | Scores |
|------:|------:|
| 2.5   | 21    |
| 5.1   | 47    |
| 7.8   | 78    |

## Dataset file location:
data/student_scores.csv


---

## 📁 Project Structure
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
│ └── student_scores.csv
│
├── outputs/
│ └── predictions.csv
│
├── templates/
│ └── index.html
│
└── static/
└── style.css


---

## ✅ Installation & Setup

**Clone the repository**
```bash
git clone https://github.com/your-username/Student-Score-Prediction.git
cd Student-Score-Prediction

** Install dependencies **
pip install -r requirements.txt

▶️ Run the Flask Web App

Start the Flask server:
python app.py
Open in browser:
http://127.0.0.1:5000/

📊 Output

The web app will show:
✅ Predicted Score (%)
✅ Graph (Hours vs Score) + Prediction point

Predictions are saved automatically in:

outputs/predictions.csv


Screenshots:

✅ Web App Home Page
✅ Prediction Result Output
✅ Graph Output (Dataset + Regression Line + Prediction Point)
📈 ML Model Used
✅ Simple Linear Regression
![alt text](image.png)

Formula:
Score = m × Hours + c

Where:
m = slope (coefficient)
c = intercept

## 👤 Author

**Anand Gopal Yadav**
📧 Email: [anandgopal2001@gmail.com](mailto:anandgopal2001@gmail.com)
🔗 LinkedIn: [https://www.linkedin.com/in/anand-gopal-yadav-223964178](https://www.linkedin.com/in/anand-gopal-yadav-223964178)
🐙 GitHub: [https://github.com/anandgopalyadav](https://github.com/anandgopalyadav)

---

⭐ If you found this project interesting, feel free to star the repository!

📜 License
This project is licensed under the MIT License.

⭐ Support
If you like this project, please ⭐ the repository!
