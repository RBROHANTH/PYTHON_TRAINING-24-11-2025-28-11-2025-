try:
    from flask import Flask
    print("Flask imported successfully!")
    
    # Create Flask application instance
    app = Flask(__name__)
    
    # Define a route for the home page
    @app.route('/')
    def home():
        return '<h1>Welcome to Flask!</h1><p>This is your first Flask application.</p>'
    
    # Define a route with a parameter
    @app.route('/user/<name>')
    def user(name):
        return f'<h1>Hello, {name}!</h1>'
    
    # Define a route for about page
    @app.route('/about')
    def about():
        return '<h1>About Page</h1><p>This is a simple Flask web application.</p>'
    
    if __name__ == '__main__':
        print("Starting Flask app...")
        print("Visit http://127.0.0.1:5000 in your browser")
        # Run the Flask app in debug mode
        app.run(debug=True, host='127.0.0.1', port=5000)
        
except ImportError as e:
    print(f"Error importing Flask: {e}")
    print("Please make sure Flask is installed: pip install flask")
except Exception as e:
    print(f"An error occurred: {e}")