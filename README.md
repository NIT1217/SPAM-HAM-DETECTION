<h1 align="center"> Spam Ham Detection Project</h1>

<p align="center">
  A Machine Learning project that detects whether a given text message is
  <b>Spam</b> or <b>Ham (Not Spam)</b> using Natural Language Processing (NLP)
  and Machine Learning techniques.
</p>

<hr>

<h2> Problem Statement</h2>

<p>
In this Data Science project, a Machine Learning system is built to detect
whether a given text message is <b>Spam</b> or <b>Ham</b>.
</p>

<ul>
  <li><b>Spam:</b> Unwanted or suspicious messages, generally sent for advertising, fraud, or malicious purposes.</li>
  <li><b>Ham:</b> Legitimate and non-spam messages.</li>
</ul>

<p>
In our daily lives, we frequently encounter spam messages through SMS,
emails, and other communication platforms. Therefore, this project aims to
build an automated system that can classify text messages as Spam or Ham.
</p>

<h2> Solution Proposed</h2>

<p>
The proposed solution uses a <b>Machine Learning-based Natural Language
Processing (NLP)</b> approach.
</p>

<p>
The system learns patterns from previously labeled text messages and uses
those patterns to predict whether a new message is Spam or Ham.
</p>

<pre>
Raw Text
   ↓
Data Preprocessing
   ↓
Text Vectorization
   ↓
Feature Extraction
   ↓
Machine Learning Model
   ↓
Prediction
   ↓
Spam / Ham
</pre>

<h2> Tech Stack Used</h2>

<ol>
  <li>Python</li>
  <li>Machine Learning</li>
  <li>Natural Language Processing (NLP)</li>
  <li>Scikit-learn</li>
  <li>Pandas</li>
  <li>NumPy</li>
  <li>Matplotlib</li>
  <li>Seaborn</li>
  <li>FastAPI</li>
  <li>Docker</li>
  <li>MongoDB</li>
</ol>

<h2> Infrastructure Required</h2>

<ul>
  <li>AWS S3</li>
  <li>AWS EC2</li>
  <li>AWS ECR</li>
  <li>GitHub Actions</li>
  <li>MongoDB Atlas</li>
</ul>

<h2> Project Structure</h2>

<pre>
Spam-Ham-Detection/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── pipeline/
│   │   └── ...
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── artifacts/
├── app.py
├── requirements.txt
├── Dockerfile
├── setup.py
├── .gitignore
└── README.md
</pre>

<h2> How to Run</h2>

<p>Before running this project, make sure you have:</p>

<ul>
  <li>Python installed</li>
  <li>Conda installed</li>
  <li>MongoDB Atlas account</li>
  <li>Git installed</li>
  <li>Docker installed (optional)</li>
</ul>

<h3>1️ Clone the Repository</h3>

<pre>
git clone &lt;YOUR_GITHUB_REPOSITORY_URL&gt;
cd Spam-Ham-Detection
</pre>

<h3> Create a Conda Environment</h3>

<h3> Install Dependencies</h3>

<pre>
pip install -r requirements.txt
</pre>

<h3> Set Environment Variables</h3>

<p><b>Windows:</b></p>

<pre>
set MONGODB_URL=&lt;YOUR_MONGODB_CONNECTION_STRING&gt;
</pre>

<pre>
export MONGODB_URL=&lt;YOUR_MONGODB_CONNECTION_STRING&gt;
</pre>

<p>
 <b>Never commit MongoDB connection strings, AWS credentials, API keys,
or other secrets to GitHub.</b>
</p>

<h2> Run the Application</h2>

<pre>
python app.py
</pre>

<p>The application will start on:</p>

<pre>
http://localhost:5000
</pre>

<h2>Training</h2>

<p>
To train the Machine Learning pipeline, open:
</p>

<pre>
http://localhost:5000/train
</pre>

<h3>Training Pipeline</h3>

<pre>
Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Pushing
</pre>

<h2> Prediction</h2>

<p>
To make predictions using the trained model:
</p>

<pre>
http://localhost:5000/predict
</pre>

<p>
The system accepts a text message and predicts whether it is:
</p>

<pre>
Spam
</pre>

<p>or</p>

<pre>
Ham
</pre>

<h2> Run Using Docker</h2>

<h3>1. Build Docker Image</h3>

<pre>
docker build \
--build-arg AWS_ACCESS_KEY_ID=&lt;AWS_ACCESS_KEY_ID&gt; \
--build-arg AWS_SECRET_ACCESS_KEY=&lt;AWS_SECRET_ACCESS_KEY&gt; \
--build-arg AWS_DEFAULT_REGION=&lt;AWS_DEFAULT_REGION&gt; \
--build-arg MONGODB_URL=&lt;MONGODB_URL&gt; \
-t spam-ham-detection .
</pre>

<h3>2. Run Docker Container</h3>

<pre>
docker run -d -p 5000:5000 spam-ham-detection
</pre>

<p>
Application URL:
</p>

<pre>
http://localhost:5000
</pre>

<h2> Machine Learning Models Used</h2>

<h3>1. Multinomial Naive Bayes</h3>

<pre>
MultinomialNB
</pre>

<p>
Suitable for text classification problems, especially when using
word-frequency-based features.
</p>

<h3>2. Gaussian Naive Bayes</h3>

<pre>
GaussianNB
</pre>

<p>
A probabilistic classification algorithm based on Bayes' theorem.
</p>

<h3>3. Support Vector Classifier</h3>

<pre>
SVC
</pre>

<p>
A powerful classification algorithm that can be used for separating
different classes in feature space.
</p>

<h2> Hyperparameter Optimization</h2>

<p>
<b>GridSearchCV</b> is used for hyperparameter optimization.
</p>

<pre>
Multiple Models
      ↓
Hyperparameter Search
      ↓
GridSearchCV
      ↓
Model Evaluation
      ↓
Best Performing Model
      ↓
Final Prediction
</pre>

<h2> NLP & Feature Extraction</h2>

<p>
Since the input data consists of text, Natural Language Processing
techniques are used to convert text into numerical features.
</p>

<h3>CountVectorizer</h3>

<p>
<b>CountVectorizer</b> converts text into numerical vectors based on the
frequency of words.
</p>

<pre>
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(lowercase=True)

X = vectorizer.fit_transform(text_data)
</pre>

<h2> ML Pipeline Components</h2>

<h3>Data Ingestion</h3>

<ul>
  <li>Loading the dataset</li>
  <li>Reading data from the required source</li>
  <li>Splitting and storing data for further processing</li>
</ul>

<h3>Data Validation</h3>

<ul>
  <li>Validating the dataset</li>
  <li>Checking data quality</li>
  <li>Checking expected columns and schema</li>
</ul>

<h3>Data Transformation</h3>

<ul>
  <li>Data preprocessing</li>
  <li>Text preprocessing</li>
  <li>Feature extraction</li>
  <li>Converting text into numerical features</li>
</ul>

<h3>Model Trainer</h3>

<ul>
  <li>Training Machine Learning models</li>
  <li>Comparing different algorithms</li>
  <li>Hyperparameter optimization</li>
  <li>Selecting the best model</li>
</ul>

<h3>Model Evaluation</h3>

<ul>
  <li>Evaluating the trained model</li>
  <li>Calculating performance metrics</li>
  <li>Comparing model performance</li>
</ul>

<h3>Model Pusher</h3>

<ul>
  <li>Saving the final model</li>
  <li>Preparing the model for deployment</li>
</ul>

<h2> Custom Exception Handling</h2>

<p>
Custom exception handling is implemented to provide meaningful error
messages and make debugging easier.
</p>

<h2> Logging</h2>

<p>
A custom logging mechanism is implemented to track:
</p>

<ul>
  <li>Pipeline execution</li>
  <li>Errors</li>
  <li>Model training</li>
  <li>Data processing</li>
  <li>Application execution</li>
</ul>

<h2> Project Workflow</h2>

<pre>
                    ┌──────────────────┐
                    │   Input Dataset  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Data Ingestion   │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Data Validation  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Data Processing  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Feature          │
                    │ Extraction       │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Model Training   │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Model Evaluation │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Best Model       │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Prediction       │
                    └──────────────────┘
</pre>

<h2> Evaluation Metrics</h2>

<ul>
  <li>Accuracy</li>
  <li>Precision</li>
  <li>Recall</li>
  <li>F1-Score</li>
  <li>Confusion Matrix</li>
</ul>

<p>
For a spam detection system, <b>Precision and Recall</b> are particularly
important because incorrectly classifying legitimate messages as spam can
negatively affect users.
</p>

<h2> Deployment</h2>

<pre>
                   GitHub
                      │
                      ↓
                GitHub Actions
                      │
                      ↓
                    AWS ECR
                      │
                      ↓
                 Docker Image
                      │
                      ↓
                   AWS EC2
                      │
                      ↓
                FastAPI App
                      │
              ┌───────┴───────┐
              ↓               ↓
         ML Model        MongoDB Atlas
</pre>

<h2> Future Improvements</h2>

<ul>
  <li>Using TF-IDF instead of simple word counts</li>
  <li>Implementing advanced NLP preprocessing</li>
  <li>Trying Logistic Regression, Random Forest, XGBoost, etc.</li>
  <li>Using Deep Learning models</li>
  <li>Adding a web-based UI</li>
  <li>Adding real-time prediction</li>
  <li>Deploying the application on AWS</li>
  <li>Adding model monitoring</li>
  <li>Implementing CI/CD using GitHub Actions</li>
</ul>

<h2> Conclusion</h2>

<p>
This project demonstrates how Machine Learning and Natural Language
Processing can be used to automatically classify text messages as
<b>Spam</b> or <b>Ham</b>.
</p>

<p>The project includes:</p>

<ul>
  <li>Data ingestion</li>
  <li>Data validation</li>
  <li>Data transformation</li>
  <li>NLP feature extraction</li>
  <li>Machine Learning model training</li>
  <li>Hyperparameter optimization</li>
  <li>Model evaluation</li>
  <li>Prediction</li>
  <li>FastAPI application</li>
  <li>Docker containerization</li>
  <li>AWS deployment support</li>
</ul>

<h2> Author</h2>

<p>
<b>Nitish Kumar</b><br>
B.Tech Artificial Intelligence<br>
SRM Institute of Science and Technology
</p>

<p align="center">
   If you found this project useful, consider giving the repository a star!
</p>
