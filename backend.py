from os.path import join

from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

import config
from captcha_v1 import evaluate


# Utils Functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

# APP
app = Flask(__name__)

app.secret_key = config.SECRET_KEY

@app.route('/', methods=['GET','POST'])
def captcha_text_v1():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash("We couldn't find your image... Please try again")
            return redirect(request.url)
        image = request.files['image']
        if image.filename == '':
            flash('Please choose an image')
            return redirect(request.url)
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(join(config.IMAGES_UPLOADED_PATH, filename))
            pred = evaluate()
            if pred[0] == "":
                pred[0] = "Sorry, I couldn't predict it"
        return render_template('captchaV1.html',prediction=pred[0])
    
    return render_template('captchaV1.html',content="Captcha prediction")

if __name__ == '__main__':
    app.run(port=1200,debug=True)
