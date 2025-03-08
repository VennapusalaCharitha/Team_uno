from flask import Flask, render_template, request

app = Flask(__name__)

# Home route to serve the index.html
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def home():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
