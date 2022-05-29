#!/usr/bin/python
#import flask.scaffold
#flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func

import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from flask import Flask
from flask_restplus import Api, Resource, fields
import joblib
from modelo import predict_proba

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Movie genre prediction',
    description='Movie Genre Prediction API')

ns = api.namespace('predict', 
     description='Genre Classifier')
   
parser = api.parser()

parser.add_argument(
    'Plot', 
    type=str, 
    required=True, 
    help='Plot', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class GenreApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_proba(args['Plot'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)