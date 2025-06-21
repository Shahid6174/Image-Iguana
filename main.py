from flask import Flask, render_template, request, flash, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge  # ✅ Correct
import cv2
import os
import numpy as np
from flask_wtf.csrf import CSRFProtect
from dotenv import load_dotenv
import hashlib

app = Flask(__name__)
csrf = CSRFProtect(app)
load_dotenv()
app.secret_key = os.environ.get('SECRET_KEY', 'dev-default-secret')  # Change this to a secure secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB limit
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    flash("File size exceeds the 2MB limit.")
    return render_template("error.html"), 413
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=True,  # Set to False during local dev without HTTPS
    SESSION_COOKIE_SAMESITE='Lax'
)


db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
class ImageMeta(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    is_deleted = db.Column(db.Boolean, default=False)
    user_id = db.Column(db.Integer)
    is_blurry = db.Column(db.Boolean, default=False)
    last_accessed = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'webp', 'png', 'jpg', 'jpeg', 'gif'}

def processImage(filename, format_conversion=None, image_processing=None):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Optional: handle format conversion here if needed

    if image_processing:
        match image_processing:
            case "cgray":
                imgProcessed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                newFilename = f"static/{filename}"
            case "histeq":
                imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imgProcessed = cv2.equalizeHist(imgGray)
                newFilename = f"static/{filename.split('.')[0]}_histeq.png"
            case "blur":
                imgProcessed = cv2.GaussianBlur(img, (5, 5), 0)
                newFilename = f"static/{filename.split('.')[0]}_blurred.png"
            case "canny":
                imgProcessed = cv2.Canny(img, 100, 200)
                newFilename = f"static/{filename.split('.')[0]}_edges.png"
            case "rotate":
                imgProcessed = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                newFilename = f"static/{filename.split('.')[0]}_rotated.png"
            case "sharpen":
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                imgProcessed = cv2.filter2D(img, -1, kernel)
                newFilename = f"static/{filename.split('.')[0]}_sharpened.png"
            case _:
                return img_path  # If unknown processing, just return original

        cv2.imwrite(newFilename, imgProcessed)
        return newFilename

    # If no processing requested, just return original path
    return img_path

@app.route("/")
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template("index.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        username = username.strip()
        email = email.strip()
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('signup'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route("/about")
@login_required
def about():
    return render_template("about.html", title="About")

def is_valid_image(file_stream):
    file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    file_stream.seek(0)  # rewind after reading
    return img is not None


@app.route("/edit", methods=["GET", "POST"])
@login_required
def edit():
    if request.method == 'POST':
        format_conversion = request.form.get("format_conversion")
        image_processing = request.form.get("image_processing")

        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return render_template("error.html")
        
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return render_template("error.html")
        elif file and allowed_file(file.filename):
            if not is_valid_image(file.stream):
               flash('Uploaded file is not a valid image.')
               return render_template("error.html")    
            file_contents = file.read()
            md5_hash = hashlib.md5(file_contents).hexdigest()
            ext = os.path.splitext(file.filename)[1]  # preserve original extension
            filename = f"{md5_hash}{ext}"

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save only if not already saved
            if not os.path.exists(file_path):
                file.seek(0)  # rewind to start after read
                file.save(file_path)

            processed_file = processImage(filename, format_conversion, image_processing)

            if processed_file:
                download_filename = os.path.basename(processed_file)
                return send_file(
                    processed_file,
                    as_attachment=True,
                    download_name=download_filename,
                    mimetype='image/png'
                )
            else:
                flash('Error processing image')
                return render_template("error.html")
        else:
            flash('File type not allowed. Please upload an image file.')
            return render_template("error.html")

    return render_template("index.html")

@app.route("/usage")
@login_required
def usage():
    return render_template("usage.html", title="Usage")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)  # Can specify the port too app.run(debug=True, port=5001)
