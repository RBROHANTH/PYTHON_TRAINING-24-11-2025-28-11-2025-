from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

users = [
    {'id': 1, 'name': 'John Smith', 'email': 'john@example.com', 'department': 'IT'},
    {'id': 2, 'name': 'Sarah Johnson', 'email': 'sarah@example.com', 'department': 'HR'},
    {'id': 3, 'name': 'Mike Davis', 'email': 'mike@example.com', 'department': 'Finance'}
]
next_id = 4

@app.route('/')
def home():
    return render_template('TEST.html')

@app.route('/home')
def home_page():
    return '<h1>Welcome to the Home Page!</h1><p>This is the home page of the Flask application.</p>'

@app.route('/users', methods=['GET'])
def display_users():
    return render_template('users.html', users=users)

@app.route('/users/create', methods=['POST'])
def create_user():
    global next_id
    new_user = {
        'id': next_id,
        'name': request.form['name'],
        'email': request.form['email'],
        'department': request.form['department']
    }
    users.append(new_user)
    next_id += 1
    return redirect(url_for('display_users'))

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)