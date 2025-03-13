from flask import Flask, render_template, request, redirect, url_for, flash, jsonify,session
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, EmailField, SelectField
from wtforms.validators import DataRequired, Length
from flask_sqlalchemy import SQLAlchemy
from flask_login import login_user, LoginManager, login_required, current_user, logout_user
import socket
import os
from ultralytics import YOLO
import cv2
import numpy as np
import base64

db = SQLAlchemy()
app = Flask(__name__)
app.config['SECRET_KEY'] = "my-secrets"
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///DeTalk.db"
db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = "modules/best.pt"
model = YOLO(model_path)

@login_manager.user_loader
def load_user(user_id):
    return Register.query.get(int(user_id))


class Register(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(50), unique=True, nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    IsBlind = db.Column(db.Integer, nullable=False)
    
    def is_active(self):
        return True

    def get_id(self):
        return str(self.id)

    def is_authenticated(self):
        return True


with app.app_context():
    db.create_all()


class RegistrationForm(FlaskForm):
    email = EmailField(label='Email', validators=[DataRequired()])
    first_name = StringField(label="First Name", validators=[DataRequired()])
    last_name = StringField(label="Last Name", validators=[DataRequired()])
    username = StringField(label="Username", validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField(label="Password", validators=[DataRequired(), Length(min=8, max=20)])
    IsBlind = SelectField('Are you blind?', choices=[('1', 'Yes'), ('0', 'No')], default='0')


class LoginForm(FlaskForm):
    email = EmailField(label='Email', validators=[DataRequired()])
    password = PasswordField(label="Password", validators=[DataRequired()])


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/login", methods=["POST", "GET"])
def login():
    form = LoginForm()
    if request.method == "POST" and form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        user = Register.query.filter_by(email=email, password=password).first()
        
        if user:
            if (session.get("IsBlind") == None):
                session['IsBlind'] = user.IsBlind
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Wrong credentials. Please try again.", "danger")  # Flashing error message for incorrect login

    return render_template("login.html", form=form)



@app.route("/logout", methods=["GET"])
@login_required
def logout():
    logout_user()
    flash("You have been logged out successfully!", "info")
    return redirect(url_for("login"))


@app.route("/register", methods=["POST", "GET"])
def register():
    form = RegistrationForm()
    if request.method == "POST" and form.validate_on_submit():
        # Check if the email or username already exists
        existing_user_email = Register.query.filter_by(email=form.email.data).first()
        existing_user_username = Register.query.filter_by(username=form.username.data).first()

        if existing_user_email:
            flash("Email already exists. Please use a different email.", "danger")
            return render_template("register.html", form=form)

        if existing_user_username:
            flash("Username already exists. Please choose a different username.", "danger")
            return render_template("register.html", form=form)
        session['IsBlind'] = form.IsBlind.data 
        # Otherwise, create the new user without hashing the password
        new_user = Register(
            email=form.email.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            username=form.username.data,
            password=form.password.data,  
            IsBlind=form.IsBlind.data 
        )
        db.session.add(new_user)
        db.session.commit()
        flash("Account created successfully! You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html", form=form)

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", first_name=current_user.first_name, last_name=current_user.last_name, IsBlind=current_user.IsBlind)


@app.route("/meeting")
@login_required
def meeting():
    return render_template("meeting.html", username=current_user.username)


@app.route("/join", methods=["GET", "POST"])
@login_required
def join():
    if request.method == "POST":
        room_id = request.form.get("roomID")
        return redirect(f"/meeting?roomID={room_id}")

    return render_template("join.html")

# API endpoint to process video frames
@app.route("/process_frame", methods=["POST"])
def process_frame():
    try:
        # Get base64-encoded image from the request
        image_data = request.json.get("frame")
        if not image_data:
            return jsonify({"error": "No frame provided"}), 400

        # Decode the image
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        image_np = np.frombuffer(binary_data, dtype=np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Run YOLO inference
        results = model.track(frame, persist=True, conf=0.5)
        annotated_frame = results[0].plot()

        # Get detected classes
        detected_classes = results[0].boxes.cls.tolist() if results[0].boxes else []
        class_names = [model.names[int(cls)] for cls in detected_classes]

        return jsonify({"transcript": " ".join(class_names)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=4000, debug=True)