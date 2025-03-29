# 🥔 Potato Disease Web App

## 🌱 Project Description
The **Potato Disease Web App** is an AI-powered tool designed to help **farmers** and **agricultural researchers** identify and understand various diseases affecting potato crops. 🏡👨‍🌾

Built with **React, FastAPI, Python, and CSS**, this web application provides an **intuitive and interactive** platform to diagnose potato diseases and suggest potential treatments. 🛠️💡 The AI model for disease detection is hosted on **Google Cloud Platform (GCP)** for seamless and scalable processing. ☁️🚀

---

## 📌 Table of Contents
- [📥 Installation](#installation)
- [🚀 Usage](#usage)
- [✨ Features](#features)
- [🤝 Contributing](#contributing)
- [📜 License](#license)
- [📧 Contact](#contact)

---

## 📥 Installation
Follow these steps to install and run the **Potato Disease Web App** locally:

1️⃣ **Clone the repository:** 📂
   ```sh
   git clone https://github.com/ShubhamJadhav03/Potato-disease-web-app.git
   ```

2️⃣ **Navigate to the project directory:** 📁
   ```sh
   cd Potato-disease-web-app
   ```

3️⃣ **Install the required dependencies:** 📦
   ```sh
   # For FastAPI (Backend dependencies)
   pip install -r requirements.txt
   
   # For React (Frontend dependencies)
   cd frontend
   npm install
   ```

---

## 🚀 Usage
Start the application by running:

1️⃣ **Run the FastAPI backend:** 🚀
   ```sh
   uvicorn main:app --reload
   ```
   The backend will be available at `http://localhost:8000`

2️⃣ **Run the React frontend:** 💻
   ```sh
   cd frontend
   npm start
   ```
   The frontend will be available at `http://localhost:3000`

3️⃣ **Google Cloud AI Model Integration:** ☁️🤖
   - The AI model for disease detection is hosted on **Google Cloud Platform (GCP)**.
   - The backend communicates with the model via **Google Cloud APIs** to process images and return predictions.
   - Ensure you have the necessary GCP credentials and API keys configured in your `.env` file.

---

## ✨ Features
✅ **AI-powered disease detection** using image processing 📷🤖  
✅ **Detailed insights** on various potato diseases 📚🔍  
✅ **Treatment recommendations** and **preventive measures** 🏥🍃  
✅ **User-friendly React interface** for seamless navigation 🎨🖥️  
✅ **FastAPI backend** for efficient data processing ⚡🔗  
✅ **Google Cloud AI model** for scalable and accurate predictions ☁️✨  

---

## 🤝 Contributing
We ❤️ contributions! If you’d like to improve this project, follow these steps:

1️⃣ **Fork the repository** 🍴  
2️⃣ **Create a new branch** ✨  
   ```sh
   git checkout -b feature/your-feature-name
   ```
3️⃣ **Make your changes and commit** 🛠️  
   ```sh
   git commit -m "Add your feature"
   ```
4️⃣ **Push your branch** 🚀  
   ```sh
   git push origin feature/your-feature-name
   ```
5️⃣ **Create a Pull Request** 🔄 and we’ll review it! 👏

---

## 📜 License
This project is **licensed under the MIT License**. See the [LICENSE](LICENSE) file for more details. ⚖️

---

## 📧 Contact
Got questions or suggestions? Reach out! 💬

👤 **Shubham Jadhav**  
🔗 **GitHub:** [ShubhamJadhav03](https://github.com/ShubhamJadhav03)  

Happy coding! 🚀🐍🎉

