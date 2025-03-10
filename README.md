# Team-UNO
# DeTalk - Debug Talk

This project is a **Flask-based web application** designed to enable seamless communication for people with disabilities, particularly those who are **visually impaired, speech impaired, or deaf**. The platform aims to **convert sign language into text and speech**, ensuring accessibility for all users.

## Features
- ✅ User authentication (Login & Registration)
- ✅ Web-based communication platform with multi-modal input and output formats
- 🚀 Sign language-to-text conversion (Planned)
- 🔌 Zoom & Google Meet Plugin (Planned)



## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/team-uno.git
cd team-uno
```



### 2. Set Up a Virtual Environment
```bash
python3 -m venv .venv  
source .venv/bin/activate   # macOS/Linux  
# OR  
.venv\Scripts\activate      # Windows  
```
### 3. Set Up a ZeGoCloud Variables
#### Change the AppId and ServerSecret variables in Meeting.html file under "Initialize Zego meeting room"

Get ZegoCloud API Keys from: https://bit.ly/3PNjNTW
- select on UI webkit for web
  
### 4. Install Dependencies
```bash
pip install -r requirements.txt

```
### 5. Go to the project directory
```bash
cd Team_uno
```

### 6. Run the Flask Application
```bash
python app.py
```
Or

###  Go to the Modules directory
```bash
cd Team_uno
cd modules
```

###  Run the Sign recognition Application
```bash
python sign.py
```

To just run the speech recognition software.

By default, the application will run on **http://127.0.0.1:5000/**. Open your browser and visit the link.

### (Optional) Deactivate Virtual Environment
After you're done, you can deactivate the virtual environment:
```bash
deactivate
```

## Future Improvements
- 🧠 AI-based sign language recognition using **MediaPipe** or **TensorFlow**
- 📢 Text-to-speech & speech-to-text support
- 📹 Integration with Zoom & Google Meet as plugins
- 🌍 Multilingual sign language support

## Contributors
👤 **Deepak Sarun Yuvachandran** | Backend/WebRTC <br>
📧 Contact: dyuvachandran@slu.edu 

👤 **Charita Vennapusala** | Frontend/UI <br>
📧 Contact: cvennapusala@slu.edu  

👤 **Jagruth Reddy Palle** | ML/Backend  <br>
📧 Contact: jpalle1@slu.edu 


