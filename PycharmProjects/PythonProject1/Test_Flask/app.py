import requests
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def say_hello():
    return render_template('home1.html')

@app.route('/add', methods=['GET'])
def add():
    try:
        a = float(request.args.get('a'))
        b = float(request.args.get('b'))
    except (TypeError, ValueError):
        return 'a и b - Не числа'
    return str(a * b)


@app.route('/user/<username>')
def show_user(username):
    return f"Hello, {username}!"

@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f"Это пост под номером: {post_id}"

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        return f"Попытка входа {username} {password}"
    else:
        return '''
                <form method = "post">
                    Логин: <input name = "username"><br>
                    Пароль: <input name = "password"><br>
                    <button type="submit">Войти</button>
                </form>'''

if __name__ == '__main__':
    app.run()