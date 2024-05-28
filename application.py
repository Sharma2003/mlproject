from flask import Flask, request, render_template
from scr.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Fetch form data and handle missing values by setting default values
            gender = request.form.get('gender', 'Not specified')
            race_ethnicity = request.form.get('ethnicity', 'Not specified')
            parental_level_of_education = request.form.get('parental_level_of_education', 'Not specified')
            lunch = request.form.get('lunch', 'Not specified')
            test_preparation_course = request.form.get('test_preparation_course', 'none')
            reading_score = float(request.form.get('reading_score', 0))
            writing_score = float(request.form.get('writing_score', 0))

            # Create CustomData object
            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score
            )

            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            predict_pipeline = PredictPipeline()
            print("Mid Prediction")
            results = predict_pipeline.predict(pred_df)
            print("After Prediction")
            return render_template('home.html', results=results[0])

        except Exception as e:
            print(f"Error occurred: {e}")
            return render_template('home.html', error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
