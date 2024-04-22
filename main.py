from functions import *
from routes import *

mkdirs()

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)