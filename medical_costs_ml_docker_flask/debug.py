from ms.functions import preprocess_data
import app
from ms.functions import get_model_response

input = [{"age":61,"sex":"female","bmi":29.1,"children":0,"smoker":"yes","region":"northwest"}]
print(input)

X = preprocess_data(input)
print(X)

response = get_model_response(X)
print(response)
